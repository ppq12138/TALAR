from ast import Not
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import os
import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
    the observations (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        policy_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


common_path = os.path.dirname(__file__)
sb3_path = os.path.dirname(common_path)
project_path = os.path.dirname(sb3_path)
envs_path = os.path.join(project_path, 'envs')
clevr_path = os.path.join(envs_path, 'clevr_robot_env')
assets_path = os.path.join(clevr_path, 'assets')


class F1(nn.Module):
    def __init__(self, input_sz, output_sz):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_sz, output_sz//2),
            nn.ReLU(),
            nn.Linear(output_sz//2, output_sz)
        )

    def forward(self, o):
        return self.layers(o)


class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.vocab = self.get_vocab()
        self.output_sz = hidden_dim
        self.embedding = nn.Embedding(len(self.vocab), self.output_sz)

    def get_vocab(self):
        with open(os.path.join(assets_path, 'vocab.txt'), 'r') as f:
            vocab_words = f.read().split('\n')
            f.close()
        # vocab_words = importlib.resources.read_text(assets, 'vocab.txt').split("\n")
        vocab_size = len(vocab_words)
        vocab = dict(zip(vocab_words, range(vocab_size)))
        return vocab

    def purify(self, text):
        return text.replace(',',' ,').replace(';',' ;').replace('?',' ?')

    def get_tokens(self, text):
        text = self.purify(text)
        return text.split()

    def tokens_to_id(self, tokens):
        ids = [self.vocab[t.lower()] for t in tokens]
        return th.LongTensor(ids).to('cuda:0')

    def forward(self, q):
        if isinstance(q, np.ndarray):
            return self._forward_batch(q)

        tokens = self.get_tokens(q)
        ids = self.tokens_to_id(tokens)

        embeddings = self.embedding(ids)
        outputs, _ = self.gru(embeddings.unsqueeze(1))

        return outputs[-1].squeeze(0)
    
    def _forward_batch(self, q): # Batch of questions
        
        tokens = [self.get_tokens(q[i]) for i in range(len(q))]

        ids = [self.tokens_to_id(tokens[i]) for i in range(len(q))]

        embeddings = [self.embedding(id_) for id_ in ids]
    
        outputs = [self.gru(embedings.unsqueeze(0))[0] for embedings in embeddings]
        outputs = [output[0][-1] for output in outputs]

        return th.stack(outputs)


def get_state_based_representation(observation, ghat, f1_model):
    '''
    Computation graph of the state-based low level policy.
    '''

    if len(observation.shape) == 2:
        observation = observation[None, :]

    observation = th.Tensor(observation)

    # Create Z Matrix
    data = []
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                data.append(th.cat((observation[i, j, :], observation[i, k, :]), 0))
    output = f1_model(th.stack(data))
    Z_matrix = output.view(observation.shape[0], observation.shape[1], observation.shape[1], -1)

    # Check for batch
    if len(ghat.shape) == 1:
        # Get Ghat
        ghat = ghat.unsqueeze(0)

    batch_size = len(Z_matrix)
    dim_1 = len(Z_matrix[0])

    # Create p matrix (Figure 8 top right matrix)
    w_matrix = th.stack([th.dot(z_vec, ghat[idx]) for idx, batch in enumerate(Z_matrix) for row in batch for z_vec in row])
    p_matrix = F.softmax(w_matrix.view(batch_size, -1), dim=1)
    p_matrix = p_matrix.view(-1, dim_1, dim_1)  

    # Create z vector
    z_vector = [[[0.0 for _ in range(5)] for _ in range(5)] for batch in observation]

    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                z_vector[i][j][k] = th.sum(p_matrix[i][j][k] * Z_matrix[i][j][k])
    
    zhat = th.stack([th.stack([th.sum(th.stack(rows)) for rows in batch]) for batch in z_vector])

    # Each o is concatenated with g and z
    state_rep = [[0.0 for _ in range(5)] for batch in observation]
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            current_o = observation[i, j, :]
            state_rep[i][j] = th.cat([current_o, ghat[i], zhat[i]], 0)

    out = th.stack([th.stack(batch) for batch in state_rep])

    return out


class LangExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        self.obs_shape = (5, 12)
        self.encode_input_dim = 64
        self.encode_output_dim = 64
        self.encoder = Encoder(self.encode_input_dim, self.encode_output_dim)
        self.f1 = F1(self.obs_shape[1] * 2, self.encoder.output_sz)
        self.extractor = nn.Flatten()

        self._features_dim = 5 * (self.obs_shape[1] + self.encoder.output_sz + 5)

    def forward(self, observations: TensorDict) -> th.Tensor:
        obs = observations['obs']
        goal = observations['goal']
        obs = obs.view((-1,) + self.obs_shape)
        goal_arr = np.array(goal)
        goal_embedding = self.encoder(goal_arr)
        zhat = get_state_based_representation(obs, goal_embedding, self.f1)

        latent = self.extractor(zhat)

        return latent


from GCP_utils.utils import total_template_list
from GCP_utils.utils import second_total_template_list
from GCP_utils.utils import total_orientation_list
from GCP_utils.utils import second_total_orientation_list
from GCP_utils.utils import color_list, color_pair2color_idx, color_idx2color_pair

from GCP_utils.language_description_for_kitchen import onehot_idx_to_description


class LangModelExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        
        forward_mode = model_kwargs.get('mode', 'low')
        is_kitchen = model_kwargs.get('is_kitchen', False)
        self.is_kitchen = is_kitchen
        self.forward_mode = forward_mode
        if self.is_kitchen:
            self.coordinate_dim = observation_space.shape[0] - 1
        else:
            self.coordinate_dim = observation_space.shape[0] - 3
        self.output_dim = self.compute_output_dim()
        self.color_arr = np.array(color_list)

    def compute_output_dim(self) -> int:
        return self._features_dim

    def features2goal(self, language_features: th.Tensor) -> list:
        language_features = language_features.long()
        batch_size = language_features.shape[0]
        language_goal_list = []
        for idx in range(batch_size):
            if self.is_kitchen:
                language_goal_idx = language_features[idx].item()
                language_goal = onehot_idx_to_description[language_goal_idx]
            else:
                template_idx = language_features[idx, 0].item()
                orientation_idx = language_features[idx, 1].item()
                color_idx = language_features[idx, 2].item()

                template = total_template_list[template_idx]
                orientation = total_orientation_list[orientation_idx]
                color_idx_arr = np.array(color_idx2color_pair[color_idx])
                color_pair = self.color_arr[color_idx_arr]

                language_goal = template.format(color_pair[0], orientation, color_pair[1])
            
            language_goal_list.append(language_goal)
        
        return language_goal_list

    def forward(self, language_features: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class OnehotLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        self.total_template_cnt = len(total_template_list)
        self.total_orientation_cnt = len(total_orientation_list)
        self.color_cnt = len(color_pair2color_idx)
                
        super().__init__(observation_space, model_kwargs)
    
    def compute_output_dim(self) -> int:
        if self.is_kitchen:
            output_dim = len(onehot_idx_to_description)
        else:
            cnt_list = [
                self.total_template_cnt,
                self.total_orientation_cnt,
                self.color_cnt,    
            ]
            
            output_dim = int(np.prod(cnt_list))
        
        return output_dim
    
    def compute_idx(self, language_features: th.Tensor) -> th.Tensor:
        if self.is_kitchen:
            idx = language_features.flatten().long()
        else:
            template_idx = language_features[..., 0]
            orientation_idx = language_features[..., 1]
            color_idx = language_features[..., 2]

            idx = template_idx * self.total_orientation_cnt * self.color_cnt\
                + orientation_idx * self.color_cnt\
                + color_idx

            idx = idx.long()
        
        return idx
    
    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode == 'low':
            batch_size = language_features.shape[0]
            onehot_idx = self.compute_idx(language_features)
            
            latent = th.zeros(batch_size, self.output_dim).to(language_features.device)
            
            bool_idx = (th.arange(batch_size).to(language_features.device), onehot_idx.to(language_features.device))
            
            latent[bool_idx] = 1
        elif self.forward_mode == 'high':
            latent = language_features.detach()
        else:
            raise NotImplementedError
        
        return latent


from algorithms.translation.sentence_encoder import FinetunedBertEncoder


class BertContLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        self.model_kwargs = model_kwargs
        
        self.bert_emb_dim = self.model_kwargs["emb_dim"]
        self.hidden_dim = model_kwargs['hidden_dim']
        self.output_dim = model_kwargs['output_dim']
        
        super().__init__(observation_space, model_kwargs)
        self.model = FinetunedBertEncoder(**model_kwargs['bert_kwargs'])
        
        self.fc = nn.Sequential(
            nn.Linear(self.bert_emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )
    
    def compute_output_dim(self) -> int:
        assert hasattr(self, 'output_dim')
        
        return self.output_dim
    
    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode in ['low', 'high']:
            language_goal_list = self.features2goal(language_features)

            bert_latent = self.model.bert_forward(language_goal_list).detach()
        elif self.forward_mode == 'extract':
            bert_latent = self.model(language_features)
        else:
            raise NotImplementedError
        
        cont_latent = bert_latent
        
        latent = self.fc(cont_latent)
        
        return latent


class BertBinaryLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        self.model_kwargs = model_kwargs
        
        self.bert_emb_dim = self.model_kwargs["emb_dim"]
        self.hidden_dim = model_kwargs['hidden_dim']
        self.output_dim = model_kwargs['output_dim']
        
        super().__init__(observation_space, model_kwargs)
        self.model = FinetunedBertEncoder(**model_kwargs['bert_kwargs'])
        
        self.fc = nn.Sequential(
            nn.Linear(self.bert_emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )
    
    def compute_output_dim(self) -> int:
        assert hasattr(self, 'output_dim')
        
        return self.output_dim
    
    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode == 'low':
            language_goal_list = self.features2goal(language_features)
        
            bert_latent = self.model.bert_forward(language_goal_list).detach()
        elif self.forward_mode == 'high':
            bert_latent = language_features.detach()
        else:
            raise NotImplementedError
        
        binary_latent = th.sigmoid(bert_latent)
        
        hard_code = binary_latent.detach().clone()
        hard_code[hard_code >= 0.5] = 1.
        hard_code[hard_code < 0.5] = 0.
        binary_latent = binary_latent + (hard_code - binary_latent).detach()
        
        latent = self.fc(binary_latent)
        
        return latent





class BaselineLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        from utils.model_utils import load_baseline_model
        
        self.encoder = load_baseline_model(
            path=model_kwargs['model_path'],
            device=model_kwargs['device'],
            kitchen=model_kwargs['is_kitchen'],
        )

        self.encoding_dim = self.compute_encoding_dim()
        self.hidden_dim = model_kwargs['hidden_dim']
        self.output_dim = model_kwargs['output_dim']

        super().__init__(observation_space, model_kwargs)

        self.fc = nn.Sequential(
            nn.Linear(self.encoding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )

    def compute_encoding_dim(self) -> int:

        encoding = self.encoder(["Random Words"])

        return encoding.shape[-1]

    def compute_output_dim(self) -> int:
        assert hasattr(self, 'output_dim')

        return self.output_dim

    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode == 'low':
            language_goal_list = self.features2goal(language_features)

            encoder_latent = self.encoder(language_goal_list).detach()
        elif self.forward_mode == 'high':
            encoder_latent = language_features.detach()
        else:
            raise NotImplementedError

        latent = self.fc(encoder_latent)

        return latent
    

class PolicyLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        from utils.model_utils import load_language_abstract_translator
        self.encoder = load_language_abstract_translator(
            path=model_kwargs['model_path'],
            device=model_kwargs['device'],
            epoch=model_kwargs['epoch'],
        )
        
        self.encoding_dim = self.compute_encoding_dim()
        self.hidden_dim = model_kwargs['hidden_dim']
        self.output_dim = model_kwargs['output_dim']
        
        super().__init__(observation_space, model_kwargs)
    
    def compute_encoding_dim(self) -> int:
        template = total_template_list[0]
        orientation = total_orientation_list[0]
        color_pair = color_list[:2]
        
        language_goal = template.format(color_pair[0], orientation, color_pair[1])
        
        encoding = self.encoder([language_goal])
        
        return encoding.shape[-1]
    
    def compute_output_dim(self) -> int:
        assert hasattr(self, 'output_dim')
        
        return self.output_dim
    
    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode in ['low', 'human']:
            language_goal_list = self.features2goal(language_features)
        
            encoder_latent = self.encoder(language_goal_list).detach()  # detach 避免梯度向后传播影响到 encoder
        elif self.forward_mode == 'high':
            encoder_latent = language_features.detach()
        else:
            raise NotImplementedError
        
        latent = encoder_latent
        
        return latent


class PolicyMLPLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        from utils.model_utils import load_mlp_translator
        self.encoder = load_mlp_translator(
            path=model_kwargs['model_path'],
            device=model_kwargs['device'],
            epoch=model_kwargs['epoch'],
        )
        
        self.encoding_dim = self.compute_encoding_dim()
        self.hidden_dim = model_kwargs['hidden_dim']
        self.output_dim = model_kwargs['output_dim']
        
        super().__init__(observation_space, model_kwargs)
        
        self.fc = nn.Sequential(
            nn.Linear(self.encoding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )

    def compute_encoding_dim(self) -> int:
        template = total_template_list[0]
        orientation = total_orientation_list[0]
        color_pair = color_list[:2]
        
        language_goal = template.format(color_pair[0], orientation, color_pair[1])
        
        encoding = self.encoder([language_goal])
        
        return encoding.shape[-1]
    
    def compute_output_dim(self) -> int:
        assert hasattr(self, 'output_dim')
        
        return self.output_dim
    
    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode == 'low':
            language_goal_list = self.features2goal(language_features)
        
            encoder_latent = self.encoder(language_goal_list).detach()  
        elif self.forward_mode == 'high':
            encoder_latent = language_features.detach()
        else:
            raise NotImplementedError
        
        latent = self.fc(encoder_latent)
        
        return latent


class PolicyComplexLMExtractor(LangModelExtractor):
    def __init__(self, observation_space: gym.Space, model_kwargs: dict = None):
        from utils.model_utils import load_language_abstract_translator
        self.encoder = load_language_abstract_translator(
            path=model_kwargs['model_path'],
            device=model_kwargs['device'],
            epoch=model_kwargs['epoch'],
        )
        
        self.encoding_dim = self.compute_encoding_dim()
        self.hidden_dim = model_kwargs['hidden_dim']
        self.output_dim = model_kwargs['output_dim']
        
        super().__init__(observation_space, model_kwargs)
        
        self.fc = nn.Sequential(
            nn.Linear(self.encoding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )
    
    def compute_encoding_dim(self) -> int:
        template = total_template_list[0]
        orientation = total_orientation_list[0]
        color_pair = color_list[:2]
        
        second_template = second_total_template_list[0]
        second_orientation = second_total_orientation_list[0]
        second_color_idx = color_list[2]
        
        first_goal = template.format(color_pair[0], orientation, color_pair[1])
        second_goal = second_template.format(second_color_idx, second_orientation)
        
        language_goal = first_goal + second_goal
        
        encoding = self.encoder([language_goal])
        
        return encoding.shape[-1]
    
    def compute_output_dim(self) -> int:
        assert hasattr(self, 'output_dim')
        
        return self.output_dim
    
    def forward(self, language_features: th.Tensor) -> th.Tensor:
        if self.forward_mode == 'low':
            language_goal_list = self.features2goal(language_features)
        
            encoder_latent = self.encoder(language_goal_list).detach()   
        elif self.forward_mode == 'high':
            encoder_latent = language_features.detach()
        else:
            raise NotImplementedError
        
        latent = self.fc(encoder_latent)
        
        return latent


class LangGCPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 language_model_type: str = 'onehot',
                 language_model_kwargs: dict = None,
                 ) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        
        if language_model_type == 'onehot':
            self.language_model = OnehotLMExtractor(observation_space, language_model_kwargs)
        elif language_model_type in ['bert_cont', 'bert']:
            self.language_model = BertContLMExtractor(observation_space, language_model_kwargs)
        # elif language_model_type == 'bert_onehot':
        #     self.language_model = BertOnehotLMExtractor(observation_space, language_model_kwargs)
        elif language_model_type == 'bert_binary':
            self.language_model = BertBinaryLMExtractor(observation_space, language_model_kwargs)
        elif language_model_type in ['policy', 'policy_ag', 'policy_binary', 'policy_cont', 'human']:
            self.language_model = PolicyLMExtractor(observation_space, language_model_kwargs)
        elif language_model_type in ['policy_mlp']:
            self.language_model = PolicyMLPLMExtractor(observation_space, language_model_kwargs)
        elif language_model_type in ['policy_complex']:
            self.language_model = PolicyComplexLMExtractor(observation_space, language_model_kwargs)
        elif language_model_type in ['baseline']:
            self.language_model = BaselineLMExtractor(observation_space, language_model_kwargs)
        else:
            raise NotImplementedError
        
        self._features_dim = self.language_model.coordinate_dim + self.language_model.output_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        observations = observations.reshape((batch_size, -1))

        coordinate_latent = observations[..., :self.language_model.coordinate_dim]
        
        language_feature = observations[..., self.language_model.coordinate_dim:]
        language_latent = self.language_model(language_feature)
        
        latent = th.cat((coordinate_latent, language_latent), dim=-1)
     
        return latent


class LangOneHotExtractor(FlattenExtractor):
    pass
  

class RecurrentGRU(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dim, rnn_layer_num):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layer_num = rnn_layer_num

        self.GRU = nn.GRU(
            input_dim,
            rnn_hidden_dim,
            rnn_layer_num,
            batch_first=True
        )

    def forward(self, x, lens, pre_hidden=None):
        if pre_hidden is None:
            pre_hidden = self.zero_hidden(batch_size=x.shape[0])
        if len(pre_hidden.shape) == 2:
            pre_hidden = th.unsqueeze(pre_hidden, dim=0)
        pre_hidden = pre_hidden.to(x.device)
        
        packed = th.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        output, hidden = self.GRU(packed, pre_hidden)
        output, _ = th.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        return output, hidden


    def zero_hidden(self, batch_size):
        return th.zeros([self.rnn_layer_num, batch_size, self.rnn_hidden_dim])


class LangRNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 vocab_size, max_seq_length,
                 output_dim=16, latent_dim=32, rnn_layer_num=1,
                 ) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.output_dim = output_dim
        self.rnn = RecurrentGRU(vocab_size, latent_dim, rnn_layer_num)
        self.fc = nn.Linear(latent_dim, output_dim)
        
        self._features_dim = get_flattened_obs_dim(observation_space) - self.max_seq_length + self.output_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        sentence_ids = observations[:, -self.max_seq_length:]
        lens = th.ones(sentence_ids.shape[0]) * self.max_seq_length
        
        sentence_ids_onehot = th.zeros((sentence_ids.shape[0], sentence_ids.shape[1], self.vocab_size)).to(observations.device)
        for idx in range(sentence_ids.shape[0]):
            sentence_ids_onehot[idx][th.arange(sentence_ids[idx].size()[0]), sentence_ids[idx].long()] = 1
        
        feat, _ = self.rnn(sentence_ids_onehot.float(), lens)
        feat = feat[:, -1, :]
        output = self.fc(feat)
        
        latent = th.cat((observations[:, :-self.max_seq_length], output), dim=1)
        
        return latent


class LangRGBExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 rgb_shape: list = [3, 64, 64],
                 rgb_output_dim: int = 64,
                 ):
        super().__init__(observation_space, features_dim=1)

        self.rgb_shape = rgb_shape
        self.rgb_size = np.prod(self.rgb_shape)
        n_input_channels = rgb_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        obs = observation_space.sample()
        rgb_arr = obs[:self.rgb_size].reshape(self.rgb_shape)
        cnn_output_dim = self.cnn(th.as_tensor(rgb_arr)).flatten().shape[0]
        
        self.rgb_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, rgb_output_dim),
            nn.ReLU(),
        )
        goal_dim = observation_space.shape[0] - self.rgb_size
        
        self._features_dim = rgb_output_dim + goal_dim

    def forward(self, observations: TensorDict) -> th.Tensor:
        goal_latent = observations[..., self.rgb_size:]
        cnn_tensor = observations[..., :self.rgb_size].reshape([-1] + self.rgb_shape)
        cnn_latent = self.cnn(cnn_tensor)
        rgb_latent = self.rgb_fc(cnn_latent)
        
        latent = th.cat([rgb_latent, goal_latent], dim=-1)
        
        return latent


class LangRGBGCPExtractor(LangGCPExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 language_model_type: str = 'onehot',
                 language_model_kwargs: dict = None,
                 rgb_shape: list = [3, 64, 64],
                 rgb_output_dim: int = 64,
                 ) -> None:
        super().__init__(observation_space, language_model_type, language_model_kwargs)

        self.rgb_shape = rgb_shape
        self.rgb_size = np.prod(self.rgb_shape)
        n_input_channels = rgb_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        obs = observation_space.sample()
        rgb_arr = obs[:self.rgb_size].reshape(self.rgb_shape)
        cnn_output_dim = self.cnn(th.as_tensor(rgb_arr)).flatten().shape[0]
        
        self.rgb_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, rgb_output_dim),
            nn.ReLU(),
        )
        self._features_dim = rgb_output_dim + self.language_model.output_dim

    def forward(self, observations: TensorDict) -> th.Tensor:
        cnn_tensor = observations[..., :self.rgb_size].reshape([-1] + self.rgb_shape)
        cnn_latent = self.cnn(cnn_tensor)
        rgb_latent = self.rgb_fc(cnn_latent)
        
        language_feature = observations[..., self.rgb_size:]
        language_latent = self.language_model(language_feature)
        
        latent = th.cat([rgb_latent, language_latent], dim=-1)
        
        return latent
    
    
def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch
