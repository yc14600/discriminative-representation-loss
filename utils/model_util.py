from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy as sp
import collections
import six
import os
import sys


from tensorflow.keras.initializers import Initializer
from utils.train_util import get_next_batch


def define_dense_layer(l,d1,d2,initialization=None,reg=None):
    w_name = 'dense_layer'+str(l)+'_weights'
    b_name = 'dense_layer'+str(l)+'_bias'

    if reg=='l2' :
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01) 
    elif reg=='l1':
        regularizer = tf.contrib.layers.l1_regularizer(scale=0.01) 
    else:
        regularizer = None

    if initialization is None:
        a = np.sqrt(3.)*np.sqrt(2./(d1+d2))

        w = tf.get_variable(name=w_name,initializer=tf.random_uniform([d1,d2],-a,a),regularizer=regularizer)

        #w = tf.get_variable(name=w_name,initializer=tf.random_normal([d1,d2],stddev=np.sqrt(3.)*np.sqrt(2./(d1+d2))),regularizer=regularizer)
        b = tf.get_variable(name=b_name,initializer=tf.zeros([d2]),regularizer=regularizer)
    else:
        W0 = initialization['w']
        if isinstance(W0, collections.Iterable):
            W0 = W0[l]
        
        B0 = initialization['b']
        if isinstance(B0, collections.Iterable):
            B0 = B0[l]

        if isinstance(W0,tf.Tensor):
            w = tf.get_variable(name=w_name,initializer=W0,regularizer=regularizer)
        else:
            w = tf.get_variable(name=w_name,shape=[d1,d2],initializer=W0,regularizer=regularizer)
            

        if isinstance(B0,tf.Tensor):
            b = tf.get_variable(name=b_name,initializer=B0,regularizer=regularizer)
            
        else:
            b = tf.get_variable(name=b_name,shape=[d2],initializer=B0,regularizer=regularizer)
            

    return w, b 


def linear(x,w,b):
    return tf.add(tf.matmul(x,w),b)


def build_dense_layer(x,l,d1,d2,initialization=None,ac_fn=tf.nn.relu,batch_norm=False,training=None,scope=None,reg=None,*args,**kargs):

    w,b = define_dense_layer(l,d1,d2,initialization,reg)
    h = linear(x,w,b)
    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/layer'+str(l))
    if ac_fn is not None:
        h = ac_fn(h)
    
    return w,b,h
    

def restore_dense_layer(x,l,w,b,ac_fn=tf.nn.relu,batch_norm=False,training=None,scope='',bayes=False,num_samples=1):
    if bayes:
        h = tf.expand_dims(x,axis=0)
        h = tf.tile(h,[num_samples,1,1])
        ew = w.sample(num_samples)
        eb = b.sample(num_samples)
        h = tf.einsum('sbi,sij->sbj',h,ew)+tf.expand_dims(eb,1)
    else:
        h = linear(x,w,b)
    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/layer'+str(l))

    if ac_fn is not None:
        h = ac_fn(linear(x,w,b))
    return h


def define_conv_layer(l,filter_shape,initialization=None,deconv=False,reg=None):
    w_name = 'conv_layer_weights'+str(l)
    b_name = 'conv_layer_bias'+str(l)

    b_shape = [filter_shape[-2]] if deconv else [filter_shape[-1]]

    if reg=='l2' :
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1) 
    elif reg=='l1':
        regularizer = tf.contrib.layers.l1_regularizer(scale=0.1) 
    else:
        regularizer = None

    if initialization is None:
        w_var = tf.get_variable(w_name,shape=filter_shape,dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0.,stddev=1e-6),regularizer=regularizer)
        b_var = tf.get_variable(b_name, b_shape, initializer=tf.constant_initializer(0.0),regularizer=regularizer)
    else:
        w0 = initialization['cw']
        b0 = initialization['cb']
        if isinstance(w0, collections.Iterable):
            w0 = w0[l]
        if isinstance(b0, collections.Iterable):
            b0 = b0[l]

        if isinstance(w0, Initializer):           
            w_var = tf.get_variable(w_name,shape=filter_shape,dtype=tf.float32,initializer=w0,regularizer=regularizer)
        else:
            w_var = tf.get_variable(w_name,dtype=tf.float32,initializer=w0,regularizer=regularizer)

        if isinstance(b0, Initializer):           
            b_var = tf.get_variable(b_name,shape=b_shape,dtype=tf.float32,initializer=b0,regularizer=regularizer)
        else:
            b_var = tf.get_variable(b_name,dtype=tf.float32,initializer=b0,regularizer=regularizer)

    return w_var,b_var


def build_conv_layer(x,l,filter_shape,strides=[1,2,2,1],padding='SAME',initialization=None,deconv=False,output_shape=None,reg=None):
    w, b = define_conv_layer(l,filter_shape,initialization,deconv,reg)
    if deconv:
        h = tf.nn.conv2d_transpose(x,filter=w,output_shape=output_shape,strides=strides,padding=padding)
    else:
        h = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding)

    h = tf.reshape(tf.nn.bias_add(h, b), h.get_shape())
    return w,b,h


def build_conv_bn_acfn(x,l,filter_shape,strides=[1,2,2,1],padding='SAME',initialization=None,deconv=False,\
                        output_shape=None,batch_norm=False,ac_fn=tf.nn.relu,training=None,scope=None,reg=None):
    print('conv layer',l,'batch norm',batch_norm,'activation',ac_fn)
    w,b,h = build_conv_layer(x,l,filter_shape,strides,padding,initialization,deconv,output_shape,reg)
    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/convlayer'+str(l))
    h = ac_fn(h)
    return w,b,h


def restore_conv_layer(x,l,w,b,strides=[1,2,2,1],padding='SAME',initialization=None,deconv=False,\
                        output_shape=None,batch_norm=False,ac_fn=tf.nn.relu,training=None,scope=''):
    
    if deconv:
        h = tf.nn.conv2d_transpose(x,filter=w,output_shape=output_shape,strides=strides,padding=padding)
    else:
        h = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding)

    h = tf.reshape(tf.nn.bias_add(h, b), h.get_shape())

    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/convlayer'+str(l))
    h = ac_fn(h)

    return h


def compute_head_output(h,w,b,output_ac=tf.nn.softmax):
    ew,eb = None, None

    if output_ac:
        h = output_ac(tf.add(tf.matmul(h,w),b))
    else:
        h = tf.add(tf.matmul(h,w),b)

    return h,ew,eb


def compute_layer_output(input,w,b,ac_fn=tf.nn.relu,head=False,num_heads=1,output_ac=tf.nn.softmax,bayes_output=False):
    h = input
    ew, eb = None, None
    if head:
        if num_heads == 1:
            h,ew,eb = compute_head_output(h,w,b,output_ac=output_ac)
        elif num_heads > 1:
            Y = []
            for wi,bi in zip(w,b):
                hi,ewi,ebi = compute_head_output(h,wi,bi,output_ac=output_ac)
                Y.append(hi)
            return Y,ew,eb
    # middle layers   
    else:
        h = ac_fn(tf.add(tf.matmul(h,w),b))

    return h,ew,eb



def forward_dense_layer(x,w,b,ac_f):
    if ac_f:
        return ac_f(tf.add(tf.matmul(x,w),b))
    else:
        return tf.add(tf.matmul(x,w),b)




def fit_model(num_iter, x_train, y_train,x_ph,y_ph,batch_size,train_step,loss,sess, print_iter=100):
    ii = 0
    for _ in range(num_iter):
            x_batch,y_batch,ii = get_next_batch(x_train,batch_size,ii,labels=y_train)
            feed_dict = {x_ph:x_batch,y_ph:y_batch}
            l,__ = sess.run([loss,train_step], feed_dict=feed_dict)
            if _% print_iter==0:
                print('loss',l)


def predict(x_test,y_test,x_ph,y,batch_size,sess,regression=False,confusion=False,feed_dict={}):
        n = int(np.ceil(x_test.shape[0]/batch_size))
        r = x_test.shape[0]%batch_size
        correct = 0.
        ii = 0
        result = []
        
        cfmtx = np.zeros([y_test.shape[1],y_test.shape[1]])
        for i in range(n):
            x_batch,y_batch,ii = get_next_batch(x_test,batch_size,ii,labels=y_test,repeat=True)
            feed_dict.update({x_ph:x_batch})
            y_pred_prob = sess.run(y,feed_dict=feed_dict)
            if i == n-1 and r>0:
                y_pred_prob = y_pred_prob[-r:]
                y_batch = y_batch[-r:]

            if len(y_pred_prob.shape) > 2:
                y_pred_prob = np.mean(y_pred_prob,axis=0)
                
            result.append(y_pred_prob)
            if not regression:
                y_pred = np.argmax(y_pred_prob,axis=1)
                correct += np.sum(np.argmax(y_batch,axis=1)==y_pred)
                if confusion: 
                    for j in range(y_test.shape[1]):
                        tot = y_batch[:,j].sum()
                        if tot > 0:
                            tmp = np.zeros_like(y_pred_prob)
                            tmp[np.arange(len(tmp)),y_pred] = 1
                            cfmtx[:,j] += tmp[y_batch[:,j]==1].sum(axis=0)

        result = np.vstack(result)  

        if not regression:  
            result = sess.run(tf.nn.softmax(result,axis=1))   
            acc = correct/y_test.shape[0]
            return acc,result,cfmtx

        else:
            return result




def square_sum_list(X):
    sum = 0.
    for x in X:
        sum += tf.reduce_sum(tf.square(x))
    return sum

def dot_prod_list(X,Y):
    sum = 0.
    for x,y in zip(X,Y):
       sum += tf.reduce_sum(x*y)
    return sum 

def mean_list(X):
    m = [0.]*len(X[0])
    for x in X:
        for xi in x:
            mi = m.pop(0)
            mi += xi
            m.append(mi)
    m = [mi/len(X) for mi in m]
    return m


def calc_similarity(vec_a, vec_b=None, sim_type='cos',sess=None):
    if isinstance(vec_a,list):
        if sim_type == 'cos':
            norm_a = tf.sqrt(square_sum_list(vec_a))
            norm_b = tf.sqrt(square_sum_list(vec_b))

            sim = dot_prod_list(vec_a,vec_b)/(norm_a*norm_b)
        elif sim_type == 'euc':
            sum = 0
            for fa,fb in zip(vec_a,vec_b):
                sum += tf.reduce_sum(tf.square(fa-fb))
            sim = tf.sqrt(sum)
    else:
        if sim_type == 'cos':
            if vec_b is not None:
                
                norm_a = tf.sqrt(tf.reduce_sum(tf.square(vec_a),axis=-1))
                norm_b = tf.sqrt(tf.reduce_sum(tf.square(vec_b),axis=-1))
                if vec_a.shape[0]!=vec_b.shape[0]:
                    sim = tf.einsum('ij,jkn->ikn',vec_a/tf.reshape(norm_a,[-1,1]),tf.transpose(tf.expand_dims(vec_b/tf.reshape(norm_b,[-1,1]),axis=1)))
                else:
                    sim = tf.matmul(vec_a/tf.reshape(norm_a,[-1,1]),tf.transpose(vec_b/tf.reshape(norm_b,[-1,1])))
            else:
                x_norm = tf.sqrt(tf.reduce_sum(tf.square(vec_a),axis=1))
                vec_a /= tf.reshape(x_norm,[-1,1])
                sim = tf.matmul(vec_a,tf.transpose(vec_a))
        elif sim_type == 'euc':
            if vec_b is not None:
                sim = tf.sqrt(tf.reduce_sum(tf.square(vec_a-vec_b)))
            else:
                k1 = tf.reshape(tf.reduce_sum(tf.square(vec_a),axis=1),[-1,1])
                k2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(vec_a),axis=1),[1,-1]),[vec_a.shape[0],1])
                sim = tf.sqrt(k1+k2-2*tf.matmul(vec_a,tf.transpose(vec_a)))

    if sess:
        sim = sess.run(sim) 

    return sim




