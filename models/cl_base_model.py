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
from abc import ABC, abstractmethod
from utils.model_util import *
from utils.train_util import *


class CL_BASE_MODEL(ABC):
    
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=512,coreset_size=0,coreset_type='random',conv=False,\
                    ac_fn=tf.nn.relu,*args,**kargs):

        self.net_shape = net_shape
        print('net shape',self.net_shape)
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.coreset_size = coreset_size
        self.coreset_type = coreset_type
        self.ac_fn = ac_fn
        self.x_ph = x_ph
        self.y_ph = y_ph
        self.conv = conv
        return


    def init_inference(self,learning_rate,train_size,decay=None,grad_type='adam',*args,**kargs):
        self.config_optimizer(starter_learning_rate=learning_rate,decay=decay,grad_type=grad_type)
        self.config_inference(train_size,*args,**kargs)

        return

    
    @abstractmethod
    def define_model(self,*args,**kargs):
        pass

    
    def config_optimizer(self,starter_learning_rate,decay=None, grad_type='adam',*agrs,**kargs):

        self.task_optimizer = config_optimizer(starter_learning_rate,'task_step',grad_type,scope='task')

        return


    @abstractmethod
    def config_inference(self,*args,**kargs):
        pass

    
    @abstractmethod
    def train_update_step(self,t,s,sess,feed_dict,kl=0.,ll=0.,err=0.,local_iter=10,*args,**kargs):
        return ll,kl,err

    
    @abstractmethod
    def train_task(self,*args,**kargs):
        pass

    @abstractmethod
    def test_all_tasks(self,*args,**kargs):
        pass


    @abstractmethod
    def update_task_data_and_inference(self,*args,**kargs):
        pass

    @abstractmethod
    def update_task_data(self,*args,**kargs):
        pass

    @abstractmethod
    def update_inference(self,*args,**kargs):
        pass


class CL_NN(CL_BASE_MODEL):
    
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=512,coreset_size=0,coreset_type='random',conv=False,dropout=None,\
                    initialization=None,ac_fn=tf.nn.relu,conv_net_shape=None,strides=None,pooling=False,\
                    coreset_mode='offline',task_type='split',*args,**kargs):
        
        super(CL_NN,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,conv,ac_fn)

        self.conv_net_shape = conv_net_shape
        self.strides = strides
        self.pooling = pooling
        print('pooling',pooling)
        self.coreset_mode = coreset_mode
        self.task_type = task_type
        print('coreset mode',self.coreset_mode,'task type',task_type)
        self.data_stats = {}

        return



    def update_data_stats(self,x_batch,y_batch):

        y_mask = np.sum(y_batch,axis=0) > 0
        nc_batch = np.sum(y_mask)                
        cls_batch = np.argsort(y_mask)[-nc_batch:]
        for c in cls_batch:
            c_stats = self.data_stats.get(c,None)
            cx = x_batch[y_batch[:,c]==1]
            if c_stats is None:
                c_stats = [np.sum(cx,axis=0), np.sum(np.square(cx),axis=0), len(cx)]
            else:
                c_stats = [np.sum(cx,axis=0)+c_stats[0], np.sum(np.square(cx),axis=1)+c_stats[1], len(cx)+c_stats[2]]
            #print('c stats',c_stats[0][c_stats[0]!=0][:5],c_stats[2])
            self.data_stats[c] = c_stats   




    def update_task_data(self,sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                        original_batch_size=500,cl_n=2,cl_k=0,cl_cmb=None,train_size=-1,test_size=-1,*args,**kargs):    
        # update data and inference for next task 
        print('train size {}, test_size {}'.format(train_size,test_size))
        if 'permuted' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=t+1,train_size=train_size,test_size=test_size)
        
        elif 'cross_split' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=t+1,cl_k=cl_k,out_dim=out_dim,train_size=train_size,test_size=test_size)
        

        elif 'split' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cl_n=cl_n,out_dim=out_dim,num_heads=self.num_heads,cl_cmb=cl_cmb,cl_k=cl_k,train_size=train_size,test_size=test_size)
        
            TRAIN_SIZE = x_train_task.shape[0]    
            if original_batch_size > TRAIN_SIZE:
                self.batch_size = TRAIN_SIZE  
            else:
                self.batch_size = original_batch_size

            print('train size',TRAIN_SIZE,'batch size',self.batch_size)


        return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss



    def update_task_data_and_inference(self,sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                                    original_batch_size=500,cl_n=2,cl_k=0,cl_cmb=None,train_size=-1,test_size=-1,*args,**kargs):    

        ## update data for next task         
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = self.update_task_data(sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                                                out_dim,original_batch_size,cl_n,cl_k,cl_cmb,train_size,test_size)
        ## update inference for next task
        self.update_inference(sess)

        return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss





