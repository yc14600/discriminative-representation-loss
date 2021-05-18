# coding: utf-8

# In[1]:


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
import time

import tensorflow as tf
import argparse
import gzip


from utils.model_util import *
from utils.train_util import *
from utils.test_util import *

from models.drl import DRL
from models.agem import AGEM
from models.multisim import MultiSim
from models.rho_margin import Rho_Margin

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets import cifar10,cifar100
import torchvision.transforms as transforms
import torch


def save_samples(path,samples,file_name=None):
    
    if not os.path.exists(path):
        os.makedirs(path)

    if file_name is None:
        file_name = ['samples','labels']
    elif not isinstance(file_name,list):
        file_name = [file_name]

    if not isinstance(samples,list):
        samples = [samples]

    for s,fname in zip(samples,file_name): 
        #print(s.shape)
        
        if not s.flags.c_contiguous:
            print('c_contiguous',s.flags.c_contiguous)
            s = np.ascontiguousarray(s)
        with gzip.open(os.path.join(path,fname+'.gz'), 'wb') as f:
            f.write(s)

    return 



parser = argparse.ArgumentParser()

parser.add_argument('-sd','--seed', default=0, type=int, help='random seed')
parser.add_argument('-ds','--dataset', default='mnist', type=str, help='specify datasets')
parser.add_argument('-dp','--data_path',default='./data/',type=str,help='path to dataset')
parser.add_argument('-rp','--result_path',default='./results/',type=str,help='the path for saving results')
parser.add_argument('-ttp','--task_type', default='split', type=str, help='task type can be split, permuted, cross split, batch')
parser.add_argument('-e','--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('-pe','--print_epoch', default=100, type=int, help='number of epochs of printing loss')
parser.add_argument('-csz','--coreset_size', default=0, type=int, help='size of each class in a coreset')
parser.add_argument('-ctp','--coreset_type', default='random', type=str, help='type of coresets')
parser.add_argument('-cmod','--coreset_mode', default='ring_buffer', type=str, help='construction mode of coresets')
parser.add_argument('-gtp','--grad_type', default='adam', type=str, help='type of gradients optimizer')
parser.add_argument('-bsz','--batch_size', default=1, type=int, help='batch size')
parser.add_argument('-trsz','--train_size', default=1000, type=int, help='size of training set')
parser.add_argument('-tesz','--test_size', default=-1, type=int, help='size of testing set')
parser.add_argument('-vdsz','--valid_size', default=0, type=int, help='size of validation set')
parser.add_argument('-nts','--num_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('-lr','--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-af','--ac_fn', default='relu', type=str, help='activation function of hidden layers')
parser.add_argument('-cltp','--cl_type', default='drl', type=str,help='cl type can be drl,agem,multisim,rho_margin')
parser.add_argument('-hdn','--hidden',default=[100,100],type=str2ilist,help='hidden units of each layer of the network')
parser.add_argument('-cv','--conv',default=False,type=str2bool,help='if use CNN on top')
parser.add_argument('-B','--B',default=3,type=int,help='training batch size')
parser.add_argument('-rg','--reg',default=None,type=str,help='regularizer')
parser.add_argument('-lrg','--lambda_reg',default=0.,type=float,help='lambda of regularizer')
parser.add_argument('-disc','--discriminant',default=False,type=str2bool,help='enable discriminant in drs cl')
parser.add_argument('-lam_dis','--lambda_disc',default=0.001,type=float,help='lambda discriminant')
parser.add_argument('-disa','--dis_alpha',default=2.,type=float,help='alpha of DRL')
parser.add_argument('-er','--ER',default='ER',type=str,help='experience replay strategy, can be ER,BER0, BER1, BER2')
parser.add_argument('-bit','--batch_iter',default=1,type=int,help='iterations on one batch')
parser.add_argument('-ntp','--net_type',default='dense',type=str,help='network type, can be dense, conv, resnet18')
parser.add_argument('-fxbt','--fixed_budget',default=True,type=str2bool,help='if budget of episodic memory is fixed or not')
parser.add_argument('-mbs','--mem_bsize',default=256,type=int,help='memory batch size used in AGEM')
parser.add_argument('-arcs','--arc_scale',default=16.,type=float,help='scale of arcface')
parser.add_argument('-arcm','--arc_margin',default=0.5,type=float,help='margin of arcface')
parser.add_argument('-mta','--mults_alpha',default=2.,type=float,help='alpha of multisim')
parser.add_argument('-mtb','--mults_beta',default=40.,type=float,help='beta of multisim')
parser.add_argument('-mtl','--mults_lamb',default=.5,type=float,help='lambda of multisim')
parser.add_argument('-st','--strength',default=100.,type=float,help='strength of auxilary objective')
parser.add_argument('-rho','--rho',default=0,type=int,help='if larger than 0, compute rho spectrum after training')
parser.add_argument('-rbe','--r_beta',default=0.6,type=float,help='init value of beta in rho_margin')
parser.add_argument('-rga','--r_gamma',default=0.2,type=float,help='gamma in rho_margin')
parser.add_argument('-rpr','--r_prob',default=0.25,type=float,help='p_rho in rho_margin')
parser.add_argument('-cuda','--cuda',default=-1,type=int,help='gpu id, -1 means no gpu')
parser.add_argument('-lam0','--lamb0',default=False,type=str2bool,help='disable L_bt in DRL')
parser.add_argument('-vf','--vis_feature',default=False,type=str2bool,help='visualize feature space')
parser.add_argument('-cmsk','--classmask',default=False,type=str2bool,help='use class mask in DRL')
parser.add_argument('-irt','--irt',default=False,type=str2bool,help='save irt response')
parser.add_argument('-irt_bi','--irt_binary_prob',default=True,type=str2bool,help='save irt response as binary')


args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    
seed = args.seed
print('seed',seed)
tf.set_random_seed(seed)
np.random.seed(seed)

ac_fn = set_ac_fn(args.ac_fn)
dataset = args.dataset

if dataset in ['mnist','fashion']:
    DATA_DIR = os.path.join(args.data_path,dataset)
else:
    DATA_DIR = args.data_path

print(dataset)

result_path = args.result_path
hidden = args.hidden 
conv = args.conv


print(args.task_type)

if 'split' in args.task_type and dataset in ['fashion','mnist','cifar10']:    
    num_tasks = 5

else:
    num_tasks = args.num_tasks


num_heads = 1
print('heads',num_heads)


# load data for different task

if  args.task_type == 'permuted':
    data = input_data.read_data_sets(DATA_DIR,one_hot=True) 
    shuffle_ids = np.arange(data.train.images.shape[0])
    X_TRAIN = data.train.images[shuffle_ids][:args.train_size]
    Y_TRAIN = data.train.labels[shuffle_ids][:args.train_size]
    X_TEST = data.test.images[:args.test_size]
    Y_TEST = data.test.labels[:args.test_size]
    out_dim = Y_TRAIN.shape[1]
    cl_n = out_dim # number of classes in each task
    cl_cmb = None
    # generate data for first task
    if 'permuted' in args.task_type:
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                                        train_size=args.train_size,test_size=args.test_size)
    else:
        x_train_task,y_train_task,x_test_task,y_test_task = X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
        clss = None

elif 'split' in args.task_type:
    if 'cifar' in dataset:
        if 'resnet18' in args.net_type:
            conv = False
            hidden = []

        else:
            conv =True
            hidden = [512,512]
        
        if dataset  == 'cifar10':
            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar10.load_data() 
            Y_TRAIN,Y_TEST = Y_TRAIN.reshape(-1), Y_TEST.reshape(-1)
            # standardize data
            X_TRAIN,X_TEST = X_TRAIN/255, X_TEST/255
            #X_TRAIN = (X_TRAIN - 0.5) * 2
            #X_TEST = (X_TEST - 0.5) * 2
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2615])
            X_TRAIN = (X_TRAIN - mean)/std
            X_TEST = (X_TEST - mean)/std
            X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
            #TRANSFORM = transforms.Compose(
            #[transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2470, 0.2435, 0.2615))])
            #X_TRAIN = TRANSFORM(torch.from_numpy(np.rollaxis(X_TRAIN,3,1))).numpy()
            #X_TEST = TRANSFORM(torch.from_numpy(np.rollaxis(X_TEST,3,1))).numpy()
            #X_TRAIN = np.rollaxis(X_TRAIN,1,3)
            #X_TEST = np.rollaxis(X_TEST,1,3)
            print('data shape',X_TRAIN.shape)
           
            if num_heads > 1:
                out_dim = 2
            else:
                out_dim = 10

            cl_cmb = np.arange(10)
            cl_k = 0
            cl_n = 2 # 2 classes per task

            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,\
                                                                        cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads) #X_TRAIN,Y_TRAIN,X_TEST,Y_TEST

        elif dataset == 'cifar100':
            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar100.load_data() 
            print(X_TRAIN.shape)
            Y_TRAIN,Y_TEST = Y_TRAIN.reshape(-1), Y_TEST.reshape(-1)
            # standardize data
            X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
            print('data shape',X_TRAIN.shape)
           
            if num_heads > 1:
                out_dim = int(100/num_tasks)
            else:
                out_dim = 100

            cl_cmb = np.arange(100)
            cl_k = 0
            cl_n = int(100/num_tasks) 
            
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,\
                                                                        cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads) 

    elif args.dataset == 'tiny_imgnet':
        from utils.tiny_imgnet import TinyImageNet
        from torch.utils.data import DataLoader
        
        if os.path.exists(os.path.join(DATA_DIR,'tiny-imagenet-200','processed')):
            
            X_TRAIN = np.load(os.path.join(DATA_DIR,'tiny-imagenet-200','processed/x_train.npy'))
            Y_TRAIN = np.load(os.path.join(DATA_DIR,'tiny-imagenet-200','processed/y_train.npy'))
            X_TEST = np.load(os.path.join(DATA_DIR,'tiny-imagenet-200','processed/x_test.npy'))
            Y_TEST = np.load(os.path.join(DATA_DIR,'tiny-imagenet-200','processed/y_test.npy'))
            print(X_TRAIN.shape,Y_TRAIN.shape)
        else:
            if os.path.exists(os.path.join(DATA_DIR,'tiny-imagenet-200')):
                download = False
            else:
                download = True
            train_dataset = TinyImageNet(DATA_DIR, split='train', download=download)
            X_TRAIN = np.vstack(list(map(lambda im: np.expand_dims(np.array(im[0]),axis=0), train_dataset)))
            Y_TRAIN = np.vstack(list(map(lambda im: np.expand_dims(np.array(im[1]),axis=0), train_dataset)))
            print(Y_TRAIN.shape)
            os.mkdir(os.path.join(DATA_DIR,'tiny-imagenet-200','processed'))
            np.save(os.path.join(train_dataset.dataset_path,'processed/x_train'), X_TRAIN)
            np.save(os.path.join(train_dataset.dataset_path,'processed/y_train'), Y_TRAIN)
            
            
            test_dataset = TinyImageNet(DATA_DIR, split='val', download=download)
            X_TEST = np.vstack(list(map(lambda im: np.expand_dims(np.array(im[0]),axis=0), test_dataset)))
            Y_TEST = np.vstack(list(map(lambda im: np.expand_dims(np.array(im[1]),axis=0), test_dataset)))
            print(X_TEST.shape)
            np.save(os.path.join(test_dataset.dataset_path,'processed/x_test'), X_TEST)
            np.save(os.path.join(test_dataset.dataset_path,'processed/y_test'), Y_TEST)
        

        Y_TRAIN,Y_TEST = Y_TRAIN.reshape(-1), Y_TEST.reshape(-1)
        # standardize data
        X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)

        out_dim = 200

        cl_cmb = np.arange(out_dim)
        cl_k = 0
        cl_n = int(out_dim/num_tasks) 
            
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,\
                                                                        cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads) 

    
    else:
        if args.dataset == 'fashion':
            url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
            data = input_data.read_data_sets(DATA_DIR,source_url=url)
        
        else:
            data = input_data.read_data_sets(DATA_DIR) 
        X_TRAIN = np.concatenate([data.train.images,data.validation.images],axis=0)
        Y_TRAIN = np.concatenate([data.train.labels,data.validation.labels],axis=0)
        X_TEST = data.test.images
        Y_TEST = data.test.labels
        if conv:
            X_TRAIN,X_TEST = X_TRAIN.reshape(-1,28,28,1),X_TEST.reshape(-1,28,28,1)
        
        if num_heads > 1:
            out_dim = 2
        else:
            out_dim = 2 * num_tasks

        cl_cmb = np.arange(10) #[0,1,2,3,4,5,6,7,8,9]#
        cl_k = 0
        cl_n = 2 #int(out_dim/args.num_tasks)
        
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,\
                                                                        test_size=args.test_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads)


TRAIN_SIZE = x_train_task.shape[0]
TEST_SIZE = x_test_task.shape[0]
original_batch_size = args.batch_size
batch_size = TRAIN_SIZE if args.batch_size > args.train_size else args.batch_size
print('batch size',batch_size)

# set results path and file name
if not os.path.exists(result_path):
    os.mkdir(result_path)

file_name = dataset+'_tsize'+str(TRAIN_SIZE)+'_cset'+str(args.coreset_size)+args.coreset_type+'_bsize'+str(batch_size)\
            +'_e'+str(args.epoch)+'_fxb'+str(args.fixed_budget)+'_'+args.task_type+'_disc'+str(args.discriminant)+'_'\
            +str(args.ER)+'_'+args.grad_type+'_'+args.cl_type+'_sd'+str(seed)

file_path = os.path.join(result_path,file_name)
file_path = config_result_path(file_path)
with open(file_path+'configures.txt','w') as f:
    f.write(str(args))


if 'resnet18' in args.net_type:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,*x_train_task.shape[1:]])
    in_dim = None
    dropout = None
    conv_net_shape,strides = None, None
    pooling = False

elif conv:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,*x_train_task.shape[1:]])
    in_dim = None
    dropout = 0.5
    if 'cifar' in dataset:
        conv_net_shape = [[3,3,3,32],[3,3,32,32],[3,3,32,64],[3,3,64,64]]
        strides = [[1,2,2,1],[1,2,2,1],[1,1,1,1],[1,1,1,1]]
        hidden = [512,256]
    else:
        conv_net_shape = [[4,4,1,32],[4,4,32,32]]
        strides = [[1,2,2,1],[1,1,1,1]]
    
    pooling = True

else:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_train_task.shape[1]])
    in_dim = x_train_task.shape[1]
    dropout = None
    conv_net_shape,strides = None, None
    pooling = False


y_ph = tf.placeholder(dtype=tf.float32,shape=[None,out_dim]) 
net_shape = [in_dim]+hidden+[out_dim]



if args.cl_type=='drl':
    Model = DRL(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,conv=conv,
            dropout=dropout,ac_fn=ac_fn,B=args.B,discriminant=args.discriminant,lambda_dis=args.lambda_disc,           
            ER=args.ER,coreset_mode=args.coreset_mode,task_type=args.task_type,batch_iter=args.batch_iter,
            net_type=args.net_type,fixed_budget=args.fixed_budget,reg=args.reg,lambda_reg=args.lambda_reg,
            alpha=args.dis_alpha,lamb0=args.lamb0,classmask=args.classmask)

elif args.cl_type=='agem':
    Model = AGEM(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,conv=conv,dropout=dropout,
            ac_fn=ac_fn,B=args.B,coreset_mode=args.coreset_mode,task_type=args.task_type,batch_iter=args.batch_iter,
            net_type=args.net_type,fixed_budget=args.fixed_budget,mem_batch_size=args.mem_bsize,reg=args.reg,lambda_reg=args.lambda_reg)


elif args.cl_type=='multisim':
    Model = MultiSim(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,conv=conv,dropout=dropout,
            ac_fn=ac_fn,B=args.B,ER=args.ER,coreset_mode=args.coreset_mode,task_type=args.task_type,batch_iter=args.batch_iter,
            net_type=args.net_type,fixed_budget=args.fixed_budget,mem_batch_size=args.mem_bsize,reg=args.reg,lambda_reg=args.lambda_reg,
            alpha=args.mults_alpha,beta=args.mults_beta,lamb=args.mults_lamb,strength=args.strength)

elif args.cl_type=='rho_margin':
    Model = Rho_Margin(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,conv=conv,dropout=dropout,
            ac_fn=ac_fn,B=args.B,ER=args.ER,coreset_mode=args.coreset_mode,task_type=args.task_type,batch_iter=args.batch_iter,
            net_type=args.net_type,fixed_budget=args.fixed_budget,mem_batch_size=args.mem_bsize,reg=args.reg,lambda_reg=args.lambda_reg,
            beta=args.r_beta,gamma=args.r_gamma,p_rho=args.r_prob,strength=args.strength)


else:
    raise TypeError('Wrong type of model')


Model.init_inference(learning_rate=args.learning_rate,decay=None,grad_type=args.grad_type)


# Start training tasks
test_sets, valid_sets = [],[]
avg_accs ,acc_record = [],[]


saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 

tf.global_variables_initializer().run(session=sess)
print('num tasks',args.num_tasks)
rho=[]
time_count = 0.
for t in range(args.num_tasks):
    # get test data
    test_sets.append((x_test_task,y_test_task))
    if args.valid_size > 0 and args.valid_size < args.train_size:
        valid_sets.append((x_train_task[-args.valid_size:],y_train_task[-args.valid_size:]))
        x_train_task, y_train_task = x_train_task[:-args.valid_size], y_train_task[:-args.valid_size]
        
    start = time.time()
    Model.train_task(sess,t,x_train_task,y_train_task,epoch=args.epoch,print_iter=args.print_epoch)
    end = time.time()

    time_count += end-start
    print('training time',time_count)

    if len(valid_sets)>0:
        print('********start validation********')
        accs, probs, _ = Model.test_all_tasks(t,valid_sets,sess,args.epoch,saver=saver,file_path=file_path,confusion=False)
    
    print('*********start testing***********')
    accs, probs, _ = Model.test_all_tasks(t,test_sets,sess,args.epoch,saver=saver,file_path=file_path,confusion=False)

    if args.irt:
        #print(probs[0].shape,test_sets[0][1].shape)
        if args.irt_binary_prob:
            probs = [np.argmax(prb,axis=1)==np.argmax(ts[1],axis=1) for prb,ts in zip(probs,test_sets)]
        else:
            probs = [prb[np.arange(len(prb)),np.argmax(ts[1],axis=1)] for prb,ts in zip(probs,test_sets)]
        for kk,pb in enumerate(probs):
            print('irt response mean',kk,pb.mean(),pb[:5])
        labels = [np.argmax(ts[1],axis=1) for ts in test_sets]

        probs = np.concatenate(probs)
        if args.irt_binary_prob:
            probs = probs.astype(np.uint8)
        #print(type(labels[0][0]))
        labels = np.concatenate(labels).astype(np.uint8)
        save_samples(file_path,[probs,labels],['test_resps_t'+str(t), 'test_labels_t'+str(t)])

    if args.vis_feature:
        tx = np.vstack([ts[0][:1000] for ts in test_sets])
        ty = np.vstack([ts[1][:1000] for ts in test_sets])
        if Model.net_type=='resnet18':
            feed_dict = {Model.training:False,Model.x_ph:tx} 
            features = sess.run(Model.H[-2],feed_dict=feed_dict) 
        else:
            feed_dict = {Model.x_ph:tx}
            features = sess.run(Model.H[:-1],feed_dict=feed_dict)
            features = np.hstack(features) 

        features_mean=features.mean(axis=0)
        features_std=features.std(axis=0)


        with open(file_path+'features_mean_'+str(t)+'.csv','w') as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(features_mean)

        with open(file_path+'features_std_'+str(t)+'.csv','w') as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(features_std)

        Model.lambda_reg = 0.
        Model.reg = None
        loss,Model.ll,Model.kl,Model.dis = Model.config_loss(Model.x_ph,Model.y_ph,Model.vars,Model.H,discriminant=Model.discriminant)
        Model.grads = tf.gradients(loss,Model.vars)
        Model.inference.config_train()

    acc_record.append(accs)
    avg_accs.append(np.mean(accs))

    if t < num_tasks-1:

        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = Model.update_task_data_and_inference(sess,t,args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                                                                                                        original_batch_size=batch_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,clss=clss,\
                                                                                                        x_train_task=x_train_task,y_train_task=y_train_task,rpath=file_path,\
                                                                                                        train_size=args.train_size,test_size=args.test_size)
    
with open(file_path+'accuracy_record.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(acc_record)):
        writer.writerow(acc_record[t])

with open(file_path+'avg_accuracy.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(avg_accs)


with open(file_path+'eplapsed_time.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow([time_count])




if args.rho > 0:
    
    print('compute rho spectrum on test sets')
    tx = np.vstack([ts[0][:200] for ts in test_sets])
    ty = np.vstack([ts[1][:200] for ts in test_sets])
    if 'resnet18' in Model.net_type:
        feed_dict = {Model.training:False,Model.x_ph:tx} 
        features = sess.run(Model.H[-2],feed_dict=feed_dict) 
    else:
        feed_dict = {Model.x_ph:tx}
        features = sess.run(Model.H[:-1],feed_dict=feed_dict)
        features = np.hstack(features)
    rho = rho_spectrum(features,mode=1)
    print('rho',rho)
    
    
    with open(file_path+'rho.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow([rho])

if args.vis_feature:
    tx = np.vstack([ts[0][:1000] for ts in test_sets])
    ty = np.vstack([ts[1][:1000] for ts in test_sets])
    if Model.net_type=='resnet18':
        feed_dict = {Model.training:False,Model.x_ph:tx} 
        features = sess.run(Model.H[-2],feed_dict=feed_dict) 
    else:
        feed_dict = {Model.x_ph:tx}
        features = sess.run(Model.H[:-1],feed_dict=feed_dict)
        features = np.hstack(features) 

    features_mean=features.mean(axis=0)
    features_std=features.std(axis=0)


    with open(file_path+'features_mean.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(features_mean)

    with open(file_path+'features_std.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(features_std)


