
# Discriminative Representation Loss

## Installation 

1. Install a virtual environment for testing, e.g.:
```
virtualenv -p python3.6 test_drl
```
2. Enable the virtual environment
```
source test_drl/bin/activate
```
3. Install required packages
```
pip install -r requirements.txt
```
    

## Reproducing results

    Please use the shell scripts for reproducing corresponding bench mark tasks, e.g. run split_mnist.sh for testing Split MNIST tasks. The default method is DRL, and please test another method by changing the first variable in .sh files, e.g.
```
METHOD_TYPE='DRL'       # method type, can be DRL, BER, ER, AGEM, MULTS, RMARGIN          
```

    GSS-greedy was tested by the source code from https://github.com/rahafaljundi/Gradient-based-Sample-Selection.


    