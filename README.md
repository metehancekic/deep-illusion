### Pytorch Adversarial Attacks #

Complete implementation of well-known attacks (PGD, FGSM, R-FGSM etc..). All attacks have apex version which you can run your attacks fast and accurate


## Module Structure #

```
project
│   README.md
│   cfo_channel_training.py     Training code for all the experiments
│   cfo_channel_testing.py      Testing code from checkpoints
│   config_cfo_channel.json     All hyper parameters for the experiment
│   simulators.py               All simulations (CFO, channel, residuals, etc) as functions
│
└───cxnn
│   │   models.py                   Neural network architectures
│   │   train.py                    Training function
│   │   train_network_reim_mag.py   Training function for real and complex networks
│   │ 
│   └───complexnn
│       │   complex-valued neural network implemantation codes
│       │   ...
│   
└───preproc   
│   │  fading_model.py      Signal processing tools (Fading models, etc)   
│   │  preproc_wifi         Preprocessing tools (Equalization, etc)
│
└───tests
    │   test_aug_analysis.py        Signal processing tools (Fading models, etc)   
    │   visualize_offset.py         Preprocessing tools (Equalization, etc)   
```
