from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
from utils.model_util import *



def define_d_net(x,net_shape,reuse,conv=False,ac_fn=tf.nn.relu,batch_norm=False,training=None,reg=None,\
                    scope = 'discriminator',pooling=False,strides=[],initialization=None,output=True,*args,**kargs):
    W,B,H = [],[],[]        
    h = x
    dense_net_shape = net_shape[1] if conv else net_shape
    
    if initialization is None:
        '''
        initialization = {'w':tf.truncated_normal_initializer(stddev=0.02),'b':tf.constant_initializer(0.0)}
        if conv:
            initialization.update({'cw':tf.random_normal_initializer(stddev=0.02),'cb':tf.constant_initializer(0.0)})
        '''
    # conv layers must before dense layers
    with tf.variable_scope(scope,reuse=reuse):
        if conv:
            for l in range(len(net_shape[0])):                    
                filter_shape = net_shape[0][l]

                strd = strides[l] if strides else [1,2,2,1]

                l_batch_norm = False
                if batch_norm and (l+1)%2 == 0:
                    l_batch_norm = True
                w,b,h = build_conv_bn_acfn(h,l,filter_shape,strides=strd,initialization=initialization,\
                                            ac_fn=ac_fn,batch_norm=l_batch_norm,training=training,scope=scope,reg=reg)
                
                W.append(w)
                B.append(b)
                H.append(h)  
            h = tf.reshape(h,[h.shape[0].value,-1])
        for l in range(len(dense_net_shape)-2):
            w, b, h = build_dense_layer(h,l,dense_net_shape[l],dense_net_shape[l+1],initialization=initialization,\
                                        ac_fn=ac_fn,batch_norm=batch_norm,training=training,scope=scope,reg=reg)
            W.append(w)
            B.append(b)
            H.append(h)  

        # define output layer without activation function  
        if output:       
            w,b,h = build_dense_layer(h,l+1,dense_net_shape[-2],dense_net_shape[-1],ac_fn=None,batch_norm=False)
            W.append(w)
            B.append(b)
            H.append(h)
        
    
    return W,B,H




