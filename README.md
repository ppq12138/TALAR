## Natural Language-conditioned Reinforcement Learning with Inside-out Task Language Development and Translation

This is official implementation of TALAR.


## Setup
1. Please finish the following steps to install conda environment and related python packages
    - Conda Environment create
    ```bash
    conda create --name <env_name> --file spec-list.txt
    ```
    - Package install
    ```bash
    pip install -r requirements.txt
    ```
2. The environments used in this work require MuJoCo, CLEVR-Robot Environment and Bert as dependecies. Please setup them following the instructions:
    - Instructions for MuJoCo: https://mujoco.org/
    - Instructions for CLEVR-Robot Environment: https://github.com/google-research/clevr_robot_env
    - Instructions for Bert: https://huggingface.co/bert-base-uncased. Move bert model to `models` directory.

## Using

We upload our full dataset in https://drive.google.com/drive/folders/1p1r5swySbafnUVfAfOCQXZiuXF3D2s3F?usp=share_link , please download the dataset before using TALAR or collect your own data.

#### Training generator of TALAR in kitchen environments:

```
python train_tl_kitchen.py
```

#### Training translator of TALAR:

```
python train_translator_kitchen.py --path <path_to_model> --cpt-epoch 0
```

#### Training goal-conditioned-policy of TALAR:

1. Finish training generator and translator of TALAR
2. Move the translator model directory to `code/models` and rename it to `code/models/policy`
```
python train_gcp.py
```

* The models of goal-conditioned-policy will be saved at `gcp_model`.
* The tensorboard log of goal-conditioned-policy will be saved at `gcp_train`.
* The evaluation result of goal-conditioned-policy will be saved at `gcp_callback`.

#### Citation

```
@inproceedings{talar/neurips/pang,
    author    = {Jing-Cheng Pang and Xinyu Yang and Si-Hang Yang and Xiong-Hui Chen andYang Yu},
    title     = {Natural Language-conditioned Reinforcement Learning with Task-related Language Development and Translation},
    booktitle = {Advances in Neural Information Processing Systems,
                {NeurIPS}},
    year      = {2023}}
```