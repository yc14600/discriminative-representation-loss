from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import csv
import copy
import six
import importlib
import os
import sys

import tensorflow as tf

from utils.model_util import *
from utils.train_util import *
from utils.resnet_util import *
from utils.net_util import define_d_net

from .drl import DRL


class Rho_Margin(DRL):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    conv=False,dropout=None,initialization=None,ac_fn=tf.nn.relu,conv_net_shape=None,strides=None,\
                    pooling=False,B=3,coreset_mode='online',batch_iter=1,task_type='split',net_type='dense',fixed_budget=True,\
                    ER=False,reg=None, lambda_reg=5e-4,beta=0.6,gamma=0.2,p_rho=0.25,strength=100.,*args,**kargs):
        
        self.beta = tf.get_variable(name='margin_beta',dtype=tf.float32,initializer=beta) 
        self.gamma = gamma
        self.p_rho = p_rho
        self.strength = strength

        super(Rho_Margin,self).__init__(net_shape=net_shape,x_ph=x_ph,y_ph=y_ph,num_heads=1,batch_size=batch_size,\
                    coreset_size=coreset_size,coreset_type=coreset_type,conv=conv,dropout=dropout,\
                    initialization=initialization,ac_fn=ac_fn,conv_net_shape=conv_net_shape,strides=strides,\
                    pooling=pooling,B=B,ER=ER,coreset_mode=coreset_mode,batch_iter=batch_iter,task_type=task_type,\
                    net_type=net_type,fixed_budget=fixed_budget,reg=reg,lambda_reg=lambda_reg,**kargs)

        return

    
    def config_loss(self,x,y,var_list,H,likelihood=True,*args,**kargs):
        
        yids = tf.matmul(y, tf.transpose(y))
        N = self.B
        mask = tf.eye(N) 
        loss = 0.

        # add beta to trainable variable list
        var_list.append(self.beta)

        
        h = self.H[-2]
        if len(h.shape) > 2:
            h = tf.reshape(h,[N,-1])
        
        h = tf.nn.l2_normalize(h,axis=1)

        if self.p_rho > 0:
            switch = np.random.choice(2,size=mask.shape,p=[1.-self.p_rho,self.p_rho])
        else:
            switch = np.zeros_like(yids)

        dists = euc_dist(h,h) 
        pos_ids = yids-mask
        neg_ids = (1.-yids)*(1-switch)+(yids-mask)*switch
        pos_dist = dists*pos_ids
        neg_dist = dists*neg_ids

        sloss = tf.reduce_sum(pos_dist-self.beta*pos_ids-neg_dist+self.beta*neg_ids+self.gamma*(1-mask))
        sloss = self.strength * sloss * 0.5
        loss+=sloss
        
                
        if self.reg:
            reg = tf.losses.get_regularization_loss()    
            loss += self.lambda_reg *reg

        if likelihood:
            #if self.task_type == 'split':
            #    loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H[-1],labels=y))
            #else:
            loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=H[-1],labels=y))

        return loss,None,None,None