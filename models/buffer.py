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




def buffer_add(t,model,x_batch,y_batch,sess=None):
            
    if model.task_type == 'split':
        y_mask = np.sum(y_batch,axis=0) > 0
        nc_batch = np.sum(y_mask)                
        cls_batch = np.argsort(y_mask)[-nc_batch:]

        for c in cls_batch:
            cx = model.core_sets.get(c,None)
            model.core_sets[c] = x_batch[y_batch[:,c]==1] if cx is None else np.vstack([cx,x_batch[y_batch[:,c]==1]])
        
    
    else:
        cxy = model.core_sets.get(t,None)
        cx = x_batch if cxy is None else np.vstack([cxy[0],x_batch])
        cy = y_batch if cxy is None else np.vstack([cxy[1],y_batch])
        model.core_sets[t] = (cx,cy)
        
    #model.online_update_coresets(model.coreset_size,model.fixed_budget,t,sess=sess)

def ring_buffer_remove(t,model,coreset_size,fixed_budget,sess=None):        

    if fixed_budget:
        clen = [(c,len(cx)) for c,cx in model.core_sets.items()] if model.task_type=='split' else [(c,len(cx[0])) for c,cx in model.core_sets.items()]
        lens = [it[1] for it in clen]
        R = np.sum(lens) - coreset_size
        while R > 0:
            ii = np.argmax(lens)
            c = clen[ii][0]
            if model.task_type == 'split':
                #model.core_sets[c] = model.core_sets[c][1:]
                kk = MD_buffer_rank(model.core_sets[c],model.data_stats[c])
                #print('kk',kk)
                model.core_sets[c] = model.core_sets[c][np.arange(lens[ii])!=kk]

            else:
                model.core_sets[c] = (model.core_sets[c][0][1:],model.core_sets[c][1][1:])
            R -= 1
            #clen = [(c,len(cx)) for c,cx in model.core_sets.items()] if model.task_type=='split' else [(c,len(cx[0])) for c,cx in model.core_sets.items()]
            #lens = [it[1] for it in clen]
            lens[ii] -= 1
        
    else:
        if model.task_type == 'split':
            for i in model.core_sets.keys():
                cx = model.core_sets[i]  
                if coreset_size < len(cx):                                                                     
                    cx = cx[-coreset_size:]
                    model.core_sets[i] = cx
                    
        else:
            ## permuted task ##
            cx = model.core_sets[t][0]
            cy = model.core_sets[t][1]
            num_per_cls = int(coreset_size/cy.shape[1])
            num_cls = np.sum(cy,axis=0).astype(int)
            
            clss = num_cls > num_per_cls
            tot = clss.sum()
            if tot > 0:
                clss = np.argsort(clss)[-tot:]
                for c in clss:
                    cids = cy[:,c]==1                            
                    rids = np.argsort(cids)[-num_cls[c]:-num_per_cls]
                    cids = np.ones(len(cx))
                    cids[rids] = 0
                    cx = cx[cids.astype(bool)]
                    cy = cy[cids.astype(bool)]
                model.core_sets[t] = (cx,cy)


def MD_buffer_rank(mem_buf,c_stats):
    f_mean = c_stats[0]/c_stats[2]#np.mean(mem_buf,axis=0)
    #print('f_mean',f_mean.shape)
    dis = np.abs(mem_buf - f_mean).sum(axis=1)
    dis_rank = np.argsort(dis)
    rank = np.arange(len(dis))+dis_rank*10
    #print('dis',dis[-5:])
    return dis_rank[np.argmin(rank)]

def MD_buffer_remove(t,model,coreset_size,fixed_budget,sess=None):

    if fixed_budget:
        clen = [(c,len(cx)) for c,cx in model.core_sets.items()] if model.task_type=='split' else [(c,len(cx[0])) for c,cx in model.core_sets.items()]
        lens = [it[1] for it in clen]
        R = np.sum(lens) - coreset_size
        while R > 0:
            ii = np.argmax(lens)
            c = clen[ii][0]
            if model.task_type == 'split':
                kk = MD_buffer_rank(model.core_sets[c],model.data_stats[c])
                model.core_sets[c] = model.core_sets[c][np.arange(lens[ii])!=kk]
            else:
                model.core_sets[c] = (model.core_sets[c][0][1:],model.core_sets[c][1][1:])
            R -= 1
            lens[ii] -= 1

        
    else:
        if model.task_type == 'split':
            for i in model.core_sets.keys():
                cx = model.core_sets[i]  
                if coreset_size < len(cx):                                                                     
                    cx = cx[-coreset_size:]
                    model.core_sets[i] = cx
                    
        else:
            ## permuted task ##
            cx = model.core_sets[t][0]
            cy = model.core_sets[t][1]
            num_per_cls = int(coreset_size/cy.shape[1])
            num_cls = np.sum(cy,axis=0).astype(int)
            
            clss = num_cls > num_per_cls
            tot = clss.sum()
            if tot > 0:
                clss = np.argsort(clss)[-tot:]
                for c in clss:
                    cids = cy[:,c]==1                            
                    rids = np.argsort(cids)[-num_cls[c]:-num_per_cls]
                    cids = np.ones(len(cx))
                    cids[rids] = 0
                    cx = cx[cids.astype(bool)]
                    cy = cy[cids.astype(bool)]
                model.core_sets[t] = (cx,cy)