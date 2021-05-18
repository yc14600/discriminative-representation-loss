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

class MultiSim(DRL):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    conv=False,dropout=None,initialization=None,ac_fn=tf.nn.relu,conv_net_shape=None,strides=None,\
                    pooling=False,B=3,coreset_mode='online',batch_iter=1,task_type='split',net_type='dense',fixed_budget=True,\
                    ER=False,reg=None, lambda_reg=5e-4,alpha=2,beta=40,lamb=0.5,strength=100.,*args,**kargs):
        self.alpha = alpha
        self.beta = beta 
        self.lamb = lamb
        self.strength = strength

        super(MultiSim,self).__init__(net_shape=net_shape,x_ph=x_ph,y_ph=y_ph,num_heads=1,batch_size=batch_size,\
                    coreset_size=coreset_size,coreset_type=coreset_type,conv=conv,dropout=dropout,\
                    initialization=initialization,ac_fn=ac_fn,conv_net_shape=conv_net_shape,strides=strides,\
                    pooling=pooling,B=B,ER=ER,coreset_mode=coreset_mode,batch_iter=batch_iter,task_type=task_type,\
                    net_type=net_type,fixed_budget=fixed_budget,reg=reg,lambda_reg=lambda_reg,**kargs)

        return

    def define_model(self,reg,*args,**kargs):
        if reg=='l2' :
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.001) 
        elif reg=='l1':
            regularizer = tf.contrib.layers.l1_regularizer(scale=0.001) 
        else:
            regularizer = None

        if self.net_type == 'dense':
    
            net_shape = [self.conv_net_shape,self.net_shape] if self.conv else self.net_shape
                
            self.qW, self.qB, H = define_d_net(self.x_ph,net_shape=net_shape,reuse=False,conv=self.conv,ac_fn=self.ac_fn,\
                                    scope='task',pooling=self.pooling,strides=self.strides,initialization=None,output=True,reg=reg)

            self.vars = self.qW+self.qB

        elif 'resnet18' in self.net_type:
            # Same resnet-18 as used in GEM paper
            self.training = tf.placeholder(tf.bool, name='train_phase')
            if self.net_type == 'resnet18_r':
                #reduced ResNet18#
                kernels = [3, 3, 3, 3, 3]
                filters = [20, 20, 40, 80, 160]
                strides = [1, 0, 2, 2, 2]
            elif self.net_type == 'resnet18_s':
                #standarded ResNet18#
                kernels = [7, 3, 3, 3, 3]
                filters = [64, 64, 128, 256, 512]
                strides = [2, 0, 2, 2, 2]
            H, self.vars = resnet18_conv_feedforward(self.x_ph,kernels=kernels,filters=filters,strides=strides,\
                                                        out_dim=self.net_shape[-1],train_phase=self.training,no_logits=False,regularizer=regularizer)
            self.qW, self.qB = [],[]

        self.H = H       

        if not self.conv:
            self.conv_W,self.conv_h = None,None
        else:
            raise NotImplementedError('Not support Conv NN yet.')

        loss = self.config_loss(self.x_ph,self.y_ph,self.vars,self.H)
        self.grads = tf.gradients(loss,self.vars) 
        self.loss = loss

    def config_loss(self,x,y,var_list,H,likelihood=True,*args,**kargs):
        
        yids = tf.matmul(y, tf.transpose(y))
        N = self.B
        mask = tf.eye(N) 
        loss = 0.
        for h in self.H:
            if len(h.shape) > 2:
                h = tf.reshape(h,[N,-1])
            
            h = tf.nn.l2_normalize(h,axis=1)
                    
            pos_sim = tf.matmul(h,tf.transpose(h))*(yids-mask)
            neg_sim = tf.matmul(h,tf.transpose(h))*(1.-yids)
            sloss = tf.divide(tf.log(1+tf.reduce_sum(tf.exp(-self.alpha*(pos_sim-self.lamb)))),self.alpha) \
                    + tf.divide(tf.log(1+tf.reduce_sum(tf.exp(self.beta*(neg_sim-self.lamb)))),self.beta)
            sloss = self.strength * sloss / N
            loss+=sloss
        
                
        if self.reg:
            reg = tf.losses.get_regularization_loss()    
            loss += self.lambda_reg *reg

        if likelihood:
            #if self.task_type == 'split':
            #    loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H[-1],labels=y))
            #else:
            loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=H[-1],labels=y))


        return loss

