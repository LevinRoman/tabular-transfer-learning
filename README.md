# Tabular Transfer Learning
The official implementation of the paper "Transfer Learning with Deep Tabular Models".

## Getting Started

### Requirements
This code was developed and tested with Python 3.8.2.

To install requirements:

```$ pip install -r requirements.txt```

## Demo Transfer Learning Experiment
While in the paper we used the MetaMIMIC test bed for our transfer learning experiments (please, see instructions below for obtaining it), we provide a demo experiment with a readily downloadable [Yeast](http://mulan.sourceforge.net/datasets-mlc.html) dataset -- a multilabel dataset with 14 targets.

We created a basic transfer learning setup by splitting the Yeast data into a multi-label [yeast_upstream](data/yeast_upstream) dataset with 13 targets for pretraining and [yeast_downstream](data/yeast_downtream) with the remaining 14-th target as the downstream target.

Now, we first pretrain FT-Transformer on the upstream data (for details please see the config files implemented using [Hydra](https://hydra.cc/docs/intro/)):

```$ python transfer_learn_net.py model=ft_transformer_pretrain dataset=yeast_upstream```

Then, we fine-tune the pretrained model on the downstream data:

```$ python transfer_learn_net.py model=ft_transformer_downstream dataset=yeast_downstream```

And compare the results to the model trained from scratch on the downstream data:

```$ python  train_net_from_scratch.py model=ft_transformer dataset=yeast_downstream```

On the pretrainining 13-target multi-label task with 1400 samples we get AUC of approximately 0.7. The model with transfer learning scores 0.63 AUC on the downstream binary task with 300 samples, while the model trained from scratch achieves 0.58 AUC.
## MetaMIMIC
In our paper we used the MetaMIMIC test bed for our transfer learning experiments which is based on the [MIMIC-IV clinical database](https://physionet.org/content/mimiciv/1.0/) of ICU admissions. Please see the [MetaMIMIC GitHub](https://github.com/ModelOriented/metaMIMIC) for instructions on constructing the MetaMIMIC dataset. Once constructed, please put it in `data/mimic/MetaMIMIC.csv` and use the provided `config/dataset/mimic.yaml` config.

## Saving Protocol 

Each time any of the main scripts are executed, a hash-like adjective-Name combination is created and saved as the `run_id` for that execution. The `run_id` is used to save checkpoints and results without being able to accidentally overwrite any previous runs with similar hyperparameters. The folder used for saving both checkpoints and results can be chosen using the following command line argument.

```$ python train_net_from_scratch.py name=<path_to_exp>```

During training, the best performing model (on held-out validation set) is saved in the folder `outputs/<path_to_exp>/training-<run_id>/model_best.pth` and the corresponding arguments for that run are saved in `outputs/<path_to_exp>/training-<run_id>/.hydra/`. 

The results are saved in `outputs/<path_to_exp>/training-<run_id>/stats.json`, the tensorboard data is saved in `outputs/<path_to_exp>/training-<run_id>/tensorboard`.

## Additional Functionality
In addition to transfer learning with deep tabular models, this repo allows to train networks from scratch using ` train_net_from_scratch.py` and to optimize their hyperparameters with [Optuna](https://optuna.org) using `optune_from_scratch.py`

## Contributing

We believe in open-source community driven software development. Please open issues and pull requests with any questions or improvements you have.

## References
* We borrow network implementations from the [RTDL repo](https://github.com/Yura52/rtdl) and extensively leverage the RTDL repo in general.
* [Yeast demo data source](http://mulan.sourceforge.net/datasets-mlc.html)
* [MetaMIMIC](https://github.com/ModelOriented/metaMIMIC)
* [MIMIC-IV clinical database](https://physionet.org/content/mimiciv/1.0/)
