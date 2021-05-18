from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def ER(t,model,buffer_size,y_batch,x_batch=None,*args,**kargs):

    clss_batch = np.sum(y_batch,axis=0) > 0
    clss_batch = np.argsort(clss_batch)[-np.sum(clss_batch):]

    if model.task_type == 'split':

        clss_mem = set(model.core_sets.keys()) - set(clss_batch)
        cx = np.vstack([model.core_sets[c] for c in clss_batch])
        mem_x = np.vstack([model.core_sets[c] for c in clss_mem])
        mem_y,cy = [],[]
        for c in clss_mem:
            tmp = np.zeros([model.core_sets[c].shape[0],model.net_shape[-1]])
            tmp[:,c] = 1
            mem_y.append(tmp)
        mem_y = np.vstack(mem_y)
        for c in clss_batch:
            tmp = np.zeros([model.core_sets[c].shape[0],model.net_shape[-1]])
            tmp[:,c] = 1
            cy.append(tmp)
        cy = np.vstack(cy)

    else:
        mem_x,mem_y,cx,cy = [],[],[],[]
        for c in model.core_sets.keys():
            if c < t:
                mem_x.append(model.core_sets[c][0])
                mem_y.append(model.core_sets[c][1])
            else:
                cx.append(model.core_sets[c][0])
                cy.append(model.core_sets[c][1])
        mem_x = np.vstack(mem_x)
        mem_y = np.vstack(mem_y)
        cx = np.vstack(cx)
        cy = np.vstack(cy)

    m_N = int(buffer_size/2)
    c_N = buffer_size-m_N
    mids = np.random.choice(mem_x.shape[0],size=m_N)
    cids = np.random.choice(cx.shape[0],size=c_N)
    coreset_x = np.vstack([cx[cids],mem_x[mids]])
    coreset_y = np.vstack([cy[cids],mem_y[mids]])

    return coreset_x,coreset_y


def BER(t,model,buffer_size,y_batch,x_batch=None,*args,**kargs):

    clss_batch = np.sum(y_batch,axis=0) > 0
    clss_batch = np.argsort(clss_batch)[-np.sum(clss_batch):]
    num_cl = len(model.core_sets)
    per_cl_size = int(buffer_size/num_cl)  
    rd = buffer_size % num_cl   
    coreset_x, coreset_y = [], []
    if model.task_type == 'split':
        if model.ER == 'BER0' and per_cl_size == 0:
            # minimum zero positive pair
            rd = rd - len(clss_batch) 
            crange = list(set(model.core_sets.keys()).difference(set(clss_batch)))
            clss = np.random.choice(crange,size=rd,replace=False)
            clss = np.concatenate([clss,clss_batch])
            per_cl_size = 1
            rd_clss = []

        elif model.ER == 'BER1' and buffer_size <= num_cl:                                             
            n_s = buffer_size - len(clss_batch) 
            crange = list(set(model.core_sets.keys()).difference(set(clss_batch)))
            clss = np.random.choice(crange,size=n_s-1,replace=False)
            clss = np.concatenate([list(clss),clss_batch])
            rd_clss = np.random.choice(list(clss),size=1,replace=False)
            rd = 1
            per_cl_size = 1

        elif model.ER == 'BER2' and per_cl_size <= 1:
            # minimum one positive pair of each selected class
            per_cl_size = 2
            crange = list(set(model.core_sets.keys()).difference(set(clss_batch)))
            clss = np.random.choice(crange,size=np.int(buffer_size/2-len(clss_batch)),replace=False)
            clss = np.concatenate([clss,clss_batch])
            rd_clss = np.random.choice(clss,size=buffer_size-len(clss)*2,replace=False) 
            rd = len(rd_clss)
            
        else:
            clss = set(model.core_sets.keys())
            rd_clss = np.random.choice(list(model.core_sets.keys()),size=rd,replace=False) if rd > 0 else [] 
        
        for i, cx in model.core_sets.items(): 
            if i in clss:
                tsize = per_cl_size+1 if rd>0 and i in rd_clss else per_cl_size
            else:
                tsize = 0
            if tsize>0:                  
                ids = np.random.choice(len(cx),size=tsize)
                tmp_y = np.zeros([tsize,model.net_shape[-1]])
                tmp_y[:,i] = 1
                tmp_x = cx[ids]
                coreset_x.append(tmp_x)
                coreset_y.append(tmp_y)
    else:
        ## permuted tasks ##
        clss = np.random.choice(list(model.core_sets.keys()),size=rd,replace=False)
        for i, cx in model.core_sets.items():
            tsize = per_cl_size+1 if rd>0 and i in clss else per_cl_size                      
            num_cl = len(model.core_sets[i][0])
            ids = np.random.choice(num_cl,size=tsize)
            tmp_x = model.core_sets[i][0][ids]
            tmp_y = model.core_sets[i][1][ids]
            coreset_x.append(tmp_x)
            coreset_y.append(tmp_y)

    return coreset_x,coreset_y