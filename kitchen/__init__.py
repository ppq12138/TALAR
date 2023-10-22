from .kitchen_envs import KitchenMicrowaveKettleLightSliderV0, KitchenMicrowaveKettleBottomBurnerLightV0
from .kitchen_envs import KitchenLanguageV0, KitchenDemoV0, KitchenLowV0, KitchenHighV0, KitchenHighNoLMV0
from gym.envs.registration import register

register(
    id='kitchen-high-nolm-v0',
    entry_point='kitchen:KitchenHighNoLMV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
    }
)

register(
    id='kitchen-high-v0',
    entry_point='kitchen:KitchenHighV0',
    max_episode_steps=28,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
        'language_model_type': 'human',
    }
)

register(
    id='kitchen-low-v0',
    entry_point='kitchen:KitchenLowV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Env upper bound exploring
register(
    id='kitchen-demo-v0',
    entry_point='kitchen:KitchenDemoV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# language
register(
    id='kitchen-language-v0',
    entry_point='kitchen:KitchenLanguageV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Smaller dataset with only positive demonstrations.
register(
    id='kitchen-complete-v0',
    entry_point='kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. A subset of the demonstrations
# solve the task.
register(
    id='kitchen-partial-v0',
    entry_point='kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. No demonstration completely
# solves the task, but each demonstration partially solves different
# components of the task.
register(
    id='kitchen-mixed-v0',
    entry_point='kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)
