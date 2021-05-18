
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

from .cl_base_model import CL_NN
from utils.model_util import *
from utils.train_util import *
from utils.resnet_util import *
from utils.net_util import define_d_net
from functools import reduce
from scipy.special import softmax
from .replay import *
from .buffer import *


class DRL(CL_NN):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    conv=False,dropout=None,initialization=None,ac_fn=tf.nn.relu,conv_net_shape=None,strides=None,\
                    pooling=False,B=3,discriminant=False,lambda_dis=.001,coreset_mode='online',batch_iter=1,task_type='split',\
                    net_type='dense',fixed_budget=True,ER='BER1',reg=None, lambda_reg=5e-4,alpha=2.,lamb0=False,\
                    classmask=False,*args,**kargs):
                    
                    
        assert(num_heads==1)
        super(DRL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    conv,dropout,initialization,ac_fn,conv_net_shape,strides,pooling,coreset_mode=coreset_mode,\
                    B=B,task_type=task_type,*args,**kargs)

        self.B = B # training batch size
        self.discriminant =discriminant
        self.lambda_dis = lambda_dis
        self.ER = ER
        self.batch_iter = batch_iter
        self.net_type = net_type
        self.fixed_budget = fixed_budget # fixed memory budget or not
        self.x_core_sets,self.y_core_sets = None, None
        self.core_sets = {}
        self.ll, self.kl = 0., 0. 
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.lamb0 = lamb0
        print('DRS_CL: B {}, ER {}, dis {}, batch iter {}, reg {}'.format(B,ER,discriminant,batch_iter,reg))
        self.classmask = classmask
        if classmask:
            self.max_c = 2
            self.c_mask = tf.placeholder(dtype=tf.float32,shape=[net_shape[-1]],name='c_mask')
        self.define_model(initialization=initialization,dropout=dropout,reg=reg)

        return


    def define_model(self,initialization=None,dropout=None,reg=None,*args,**kargs):

        if self.net_type == 'dense':

            net_shape = [self.conv_net_shape,self.net_shape] if self.conv else self.net_shape
                
            self.qW, self.qB, self.H = define_d_net(self.x_ph,net_shape=net_shape,reuse=False,conv=self.conv,ac_fn=self.ac_fn,\
                                    scope='task',pooling=self.pooling,strides=self.strides,initialization=initialization,reg=reg)
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
            if reg=='l2' :
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.01) 
            elif reg=='l1':
                regularizer = tf.contrib.layers.l1_regularizer(scale=0.01) 
            else:
                regularizer = None
            self.H, self.vars = resnet18_conv_feedforward(self.x_ph,kernels=kernels,filters=filters,strides=strides,
                                                        out_dim=self.net_shape[-1],train_phase=self.training,regularizer=regularizer)
            self.qW, self.qB = [],[]
            print(np.sum([np.prod([s.value for s in v.shape]) for v in self.vars]),[h.shape for h in self.H])
        if not self.conv:
            self.conv_W,self.conv_h = None,None
        else:
            raise NotImplementedError('Not support Conv NN yet.')


        loss,self.ll,self.kl,self.dis = self.config_loss(self.x_ph,self.y_ph,self.vars,self.H,discriminant=self.discriminant)
        self.grads = tf.gradients(loss,self.vars)

        
    
    def init_inference(self,learning_rate,decay=None,grad_type='adam',*args,**kargs):
        self.config_optimizer(starter_learning_rate=learning_rate,decay=decay,grad_type=grad_type)
        self.config_inference(*args,**kargs)

        return

    

    def config_inference(self,*args,**kargs):

        self.inference = MLE_Inference(var_list=self.vars,grads=self.grads,optimizer=self.task_optimizer,ll=self.ll,kl=self.kl)

    
    
    def config_loss(self,x,y,var_list,H,discriminant=True,likelihood=True,compact_center=False,*args,**kargs):
        loss,ll,reg, dis = 0.,0.,0.,0.
        
        if likelihood:
            
            if not self.classmask:
                ll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=H[-1],labels=y))
            else:
                logits = H[-1] + self.c_mask
                ll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))
            
            loss += ll

        if discriminant:
            yids = tf.matmul(y, tf.transpose(y))
            N = self.B
            mask = tf.eye(N) 

            for h in H[:]:                
                if len(h.shape) > 2:
                    h = tf.reshape(h,[N,-1])
                #h = tf.nn.l2_normalize(h,axis=1)

                sim = tf.matmul(h,tf.transpose(h))
                if not self.lamb0:
                    dis += tf.reduce_mean(sim*(1.-yids)+self.alpha*sim*(yids-mask))
                else:
                    dis += tf.reduce_mean(self.alpha*sim*(yids-mask))
                    
            loss += self.lambda_dis * dis 

        if self.reg:
            print('add regularization loss')
            reg = tf.losses.get_regularization_loss()    
            loss += self.lambda_reg *reg

        return loss,ll,reg,dis
   

    def update_train_batch(self,t,s,sess,feed_dict,*args,**kargs):
        y_batch = feed_dict[self.y_ph]
        buffer_size = self.B 
        if t > 0:

            if self.ER == 'ER':

                coreset_x, coreset_y = ER(t,self,buffer_size,y_batch)
                
            else:
                ###### BER #####
                coreset_x,coreset_y = BER(t,self,buffer_size,y_batch)

                    
            if isinstance(coreset_x,list):
                coreset_x, coreset_y = np.vstack(coreset_x), np.vstack(coreset_y)
            feed_dict.update({self.x_ph:coreset_x,self.y_ph:coreset_y})  
           

        ### first task ###              
        else:
            if self.task_type == 'split':
                cx, cy = [], []
                for c in self.core_sets.keys():
                    cx.append(self.core_sets[c])
                    tmp_y = np.zeros([cx[-1].shape[0],self.net_shape[-1]])
                    tmp_y[:,c] = 1
                    cy.append(tmp_y)

                cx = np.vstack(cx)
                cy = np.vstack(cy)
                cx, cy = shuffle_data(cx,cy)
            else:
                cx, cy = self.core_sets[t] 

            bids = np.random.choice(len(cx),size=buffer_size) 
            feed_dict.update({self.x_ph:cx[bids],self.y_ph:cy[bids]})

        if self.task_type=='split' and self.classmask: 
            c_mask = np.zeros(self.net_shape[-1])
            c_mask[self.max_c:] = -np.inf
            feed_dict.update({self.c_mask:c_mask})  

        return feed_dict


    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)

        x_batch, y_batch = feed_dict[self.x_ph], feed_dict[self.y_ph]

        if local_iter == 0:  
            self.update_buffer(t,x_batch,y_batch,sess=sess)
        if self.classmask:    
            self.max_c = max(self.core_sets.keys()) +1   
        feed_dict = self.update_train_batch(t,s,sess,feed_dict)

        self.inference.update(sess=sess,feed_dict=feed_dict)

        return err


    def train_task(self,sess,t,x_train_task,y_train_task,epoch,print_iter=5,\
                    tfb_merged=None,tfb_writer=None,tfb_avg_losses=None,*args,**kargs):

        # training for current task
        num_iter = int(np.ceil(x_train_task.shape[0]/self.batch_size))
        
        for e in range(epoch):
            shuffle_inds = np.arange(x_train_task.shape[0])
            np.random.shuffle(shuffle_inds)
            x_train_task = x_train_task[shuffle_inds]
            y_train_task = y_train_task[shuffle_inds]
            err = 0.
            ii = 0
            for _ in range(num_iter):
                x_batch,y_batch,ii = get_next_batch(x_train_task,self.batch_size,ii,labels=y_train_task)

                for __ in range(self.batch_iter):
                    feed_dict = {self.x_ph:x_batch,self.y_ph:y_batch}
                    if 'resnet18' in self.net_type:
                        feed_dict.update({self.training:True})

                    err = self.train_update_step(t,_,sess,feed_dict,err,x_train_task,y_train_task,local_iter=__,*args,**kargs)
            if (e+1)%print_iter==0:
                if self.discriminant:
                    ll,kl,dis = sess.run([self.ll,self.kl,self.dis],feed_dict=feed_dict)
                    print('epoch',e+1,'ll',ll,'kl',kl,'dis',dis)

        return


    def update_inference(self,sess,*args,**kargs):
        self.inference.reinitialization(sess)
        return


    def update_buffer(self,t,x_batch,y_batch,sess=None):

        buffer_add(t,self,x_batch,y_batch,sess=sess)
        self.update_data_stats(x_batch,y_batch)
        self.online_update_coresets(t,sess=sess)
        
        return
    

    def online_update_coresets(self,t,sess=None):
    
        if self.coreset_mode == 'ring_buffer':
            ring_buffer_remove(t,self,self.coreset_size,self.fixed_budget,sess=sess)
        else:
            raise NotImplementedError('Not implememted buffer type.')


    def test_all_tasks(self,t,test_sets,sess,epoch=10,saver=None,file_path=None,confusion=False,*args,**kargs):
        acc_record, pred_probs = [], []
        dim = test_sets[0][1].shape[1]
        cfmtx = np.zeros([dim,dim])
        feed_dict = {self.training:False} if 'resnet18' in self.net_type else {}

        for t,ts in enumerate(test_sets): 
            acc, y_probs,cfm = predict(ts[0],ts[1],self.x_ph,self.H[-1],self.batch_size,sess,regression=False,confusion=confusion,feed_dict=feed_dict)
            print('accuracy',acc)
            acc_record.append(acc)
            pred_probs.append(y_probs)
            cfmtx += cfm
        print('avg accuracy',np.mean(acc_record))
        return acc_record,pred_probs,cfmtx


class MLE_Inference:
    def __init__(self,var_list,grads,optimizer=None,ll=0.,kl=0.,*args,**kargs):
        self.var_list = var_list
        self.grads = grads
        self.optimizer = optimizer
        self.ll = ll
        self.kl = kl
        self.config_train()

    def reinitialization(self,sess=None,scope='task',warm_start=True,*args,**kargs):
        if not warm_start:
            reinitialize_scope(scope=scope,sess=sess)
        return

    
    def config_train(self,*args,**kargs):
        
        grads_and_vars = list(zip(self.grads,self.var_list))
        self.train = self.optimizer[0].apply_gradients(grads_and_vars,global_step=self.optimizer[1])

        return

    
    def update(self,sess,feed_dict=None,*args,**kargs):

        sess.run(self.train, feed_dict)

        return
    


