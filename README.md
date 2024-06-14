# Multi-Task-Learning-PyTorch

This code repository is heavily based on [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch) repository.

## Usage

### Setup 
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/pascal/`.
- Specify the output directory in `configs/your_env.yml`. All results will be stored under this directory.

### Training
The configuration files to train the model can be found in the `configs/` directory. The model can be trained by running the following command:

```shell
python main.py --config_env configs/env.yml --config_exp configs/$DATASET/$MODEL.yml
```

### Distributed Training
The code can be run on multiple GPUs using the following command:
```shell
bash dist_train.sh configs/$DATASET/$MODEL.yml $NUM_GPUS
```