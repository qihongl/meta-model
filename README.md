# LCNet-meta-modeling
This is the repo for the following paper:

Lu, Q., Nguyen, T. T., Zhang, Q., Hasson, U., Griffiths, T. L., Zacks, J. M., Gershman, S. J., & Norman, K. A. (2023). **Reconciling Shared versus Context-Specific Information in a Neural Network Model of Latent Causes.** In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/2312.08519

This repo contains the code for Simulation 3 using the [META data](https://psyarxiv.com/pt6hx/).   

### code dependency 

In additional to the dependencies related to the META data ([here](https://github.com/mbezdek/extended-event-modeling)), a reasonably up-to-date version of pytorch is needed. 

### to replicate the results: 

To replicate the results, a computing cluster is needed. We used [slurm](https://slurm.schedmd.com/documentation.html). 

On the cluster, run the following code under `src`
```sh
./submit-train.sh
```
Once model training is done, run the following code under `src`
```sh
./submit-eval-group.sh 
```
Once model evaluation is done, you can find the results under the logging directory you specified in `eval-group.sh` (i.e., [here](https://github.com/qihongl/meta-model/blob/main/src/eval-group.sh#L9C13-L9C13)).

### directory structure
```sh
src
├── demo
│   └── ...... 
├── slurm_log
│   └── ...... 
├── utils                         # utils used to organize META data 
│   ├── DataLoader.py
│   ├── EventLabel.py
│   ├── HumanBondaries.py
│   ├── PCATransformer.py
│   ├── Parameters.py
│   ├── SegmentationVideo.py
│   ├── TrainValidFMRISplit.py
│   ├── TrainValidSplit.py
│   ├── _METAConstants.py
│   ├── _METAVideos.py
│   └── __init__.py
├── model                        # model components 
│   ├── CGRU.py
│   ├── CGRU_c2h.py
│   ├── CGRU_v2.py
│   ├── NNShortCut.py
│   ├── SimpleContext.py
│   ├── SimpleMemory.py
│   ├── SimpleShortcut.py
│   ├── SimpleTracker.py
│   ├── SimpleUniformContext.py
│   ├── TabularShortCutIS.py
│   ├── Vanilla_iSITH.py
│   └── __init__.py
├── train-lcn.py          # model training script 
├── train.sh              # used on cluster: submit the model training script with a particular set of hyperparameters 
├── submit-train.sh       # used on cluster: submit a batch of model training scripts with a grid of hyperparameters
├── eval-group.py         # model evaluation script 
├── eval-group.sh         # used on cluster: submit the model evaluation script with a particular set of hyperparameters 
└── submit-eval-group.sh  # used on cluster: submit a batch of model evaluation scripts with a grid of hyperparameters
```

### links to the META data

META data: https://osf.io/3f9d2/

META data paper: https://psyarxiv.com/r5tju/ 

SEM OSF: https://osf.io/39qwz/

SEM paper: https://psyarxiv.com/pt6hx/

repo: https://github.com/mbezdek/extended-event-modeling


