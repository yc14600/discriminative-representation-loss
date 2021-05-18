import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os

from scipy.stats import multivariate_normal, norm
from utils.train_util import gen_class_split_data,one_hot_encoder,shuffle_data




def str2bool(x):
    if x.lower() == 'false':
        return False
    else:
        return True

def str2ilist(s):   
    s = s[1:-1]
    s = s.split(',')
    l = [int(si) for si in s]
    return l

def str2flist(s):   
    s = s[1:-1]
    s = s.split(',')
    l = [float(si) for si in s]
    return l

def normalize(x):
    s = np.sum(x)
    y = [xi/s for xi in x]
    return y



def config_result_path(rpath):
    if rpath[-1] != '/':
        rpath = rpath+'/'

    try:
        os.makedirs(rpath)
    except FileExistsError: 
        return rpath
    
    return rpath

def rho_spectrum(features,mode=1):
    from sklearn.decomposition import TruncatedSVD
    from scipy.stats import entropy
    embed_dim = features.shape[-1]
    svd = TruncatedSVD(n_components=embed_dim-1, n_iter=7, random_state=42)
    svd.fit(features)

    s = svd.singular_values_
    s = s[np.abs(mode)-1:]
    s_norm  = s/np.sum(s)
    uniform = np.ones(len(s))/(len(s))
    kl = entropy(uniform, s_norm)

    return kl

