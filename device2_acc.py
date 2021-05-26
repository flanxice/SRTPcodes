# -*- coding: utf-8 -*-
#信道分配和发送功率已知，优化MPSK的M
import numpy as np
from multiprocessing.connection import Client
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import scipy.io
from models import *
import time
import math
from numpy import random





device_index = 2

address = ('localhost', 6000+device_index)
conn = Client(address,authkey=b"password")

gpu_id = conn.recv()
print(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

device_num = int(conn.recv())
net_choice = conn.recv()  # recvive net_choice
LR = float(conn.recv())  # receive learning rate
batchsize = conn.recv()  # get batchsize and compress rate
compress_rate = conn.recv()

# net_choice = 'alexnet'



if net_choice == "alexnet":
    pic_size = 252
    myNN = torchvision.models.alexnet(num_classes=10)
elif net_choice == "vgg":
    pic_size = 224
    myNN = torchvision.models.vgg11(num_classes=10)
elif net_choice == "inception":
    pic_size = 299
    myNN = torchvision.models.inception_v3(num_classes=10, aux_logits=False)
elif net_choice == "resnet":
    pic_size = 32
    myNN = ResNet18()
elif net_choice == "densenet":
    pic_size = 32
    myNN = DenseNet121()
elif net_choice == "mobilenet":
    pic_size = 32
    myNN = MobileNetV2()
else:
    pass
myNN = myNN.cuda()
myNN = nn.DataParallel(myNN)
# load data
data_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
data_transform_train = transforms.Compose([
        #transforms.Resize(pic_size),
        transforms.ToPILImage(),
        transforms.RandomCrop(pic_size, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

# data number
data_num = int(50000/device_num)
# shuffled index for user
shuffled_index = scipy.io.loadmat('./shuffled_index.mat')['shuffled_index'].tolist()[0]
shuffled_index_user = shuffled_index[data_num*(device_index-1):data_num*device_index]
# blank train data&label to be filled
train_data = torch.zeros([data_num, 3, pic_size, pic_size])
train_label = torch.zeros([data_num], dtype=torch.long)
for index, [data,label] in enumerate(trainloader):
    if index in shuffled_index_user:
        data_position = shuffled_index_user.index(index)
        train_data[data_position] = data
        train_label[data_position] = label


myNN.load_state_dict(torch.load("./server_model_acc/server_model_initial.pkl"))  # load initial model


optimizer = torch.optim.SGD(myNN.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
data_index = 0
iteration_index = 1
calc_grad_time_list_user = []
update_grad_time_list_user = []



conn.send('prepare work done '+str(device_index))

while True:   # 分配 batch data, normalize x when iterate train_loader

    BATCHSIZE = int(batchsize)
    # if exit
    if BATCHSIZE == -1:
        break
    else:
        pass
    COMPRESS = float(compress_rate)


    conn.recv()#''synchronization''
    conn.send('prepare download')
    myNN.load_state_dict(torch.load("./server_model_acc/server_model.pkl"))  # load model (download parameter)
    # optimizer.zero_grad()  # load server_gradient (download gradient)
    # server_gradient = torch.load('./grads/server_grads.pkl')
    # for index, (name, param) in enumerate(myNN.named_parameters()):
    #     param.grad = server_gradient[index]
    # optimizer.step()  # update parameter then train

    # make batch data
    if data_index + BATCHSIZE > data_num:
        data_index = 0
    else:
        pass
    
    x_temple = train_data[data_index: data_index + BATCHSIZE]
    y = train_label[data_index: data_index + BATCHSIZE]
    x = copy.deepcopy(x_temple)
    for i in range(BATCHSIZE):
        x[i] = data_transform_train(x_temple[i])
    x = x.cuda()
    y = y.cuda()
    data_index = data_index + BATCHSIZE

    output = myNN(x)
    loss = loss_func(output, y)
    myNN.zero_grad()
    loss.backward()
    names_params = list(myNN.named_parameters())  # compress and save gradient
    grads = []
    for (name, param) in names_params:
        grads_temple = param.grad
        grads_one = torch.abs(torch.reshape(grads_temple, (1, -1)))  # compress gradient
        grad_num = grads_one.size()[1]
        
        if int(grad_num / COMPRESS) == 0:
            threshold = 0
        else:
            threshold_list = np.reshape(torch.topk(grads_one, int(grad_num / COMPRESS), sorted=True, largest=True)[0].tolist(), (-1, ))
            threshold = threshold_list[-1]
        gradient=torch.where(torch.abs(grads_temple) < threshold, torch.zeros_like(grads_temple), grads_temple)#compress之后的gradient
        #gradient=pe.packet_error(Bu,Bd,bits_num,distance,pd,N0,snr,M,gradient)
        grads.append(gradient)
    
   
    torch.save(grads, './grads_acc/grads'+str(device_index)+'.pkl')
    torch.save(loss, './loss_acc/loss'+str(device_index)+'.pkl')
    conn.send('aaaa')

conn.close()

