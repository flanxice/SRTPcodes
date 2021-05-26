from multiprocessing.connection import Listener
import os
import numpy as np
import scipy.io
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import *

# import problem_opt

device_num = 3
print('---------create connect--------------')
address1 = ('localhost', 6001)
listener1 = Listener(address1, authkey=b"password")
address2 = ('localhost', 6002)
listener2 = Listener(address2, authkey=b"password")
address3 = ('localhost', 6003)
listener3 = Listener(address3, authkey=b"password")
# address4 = ('localhost', 6004)
# listener4 = Listener(address4, authkey=b"password")
# address5 = ('localhost', 6005)
# listener5 = Listener(address5,authkey=b"password")

gpu_id = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

if not os.path.exists("./server_model_acc/"):
    os.makedirs("./server_model_acc/")
if not os.path.exists("./log_acc/"):
    os.makedirs("./log_acc/")
if not os.path.exists("./grads_acc/"):
    os.makedirs("./grads_acc/")
if not os.path.exists("./loss_acc/"):
    os.makedirs("./loss_acc/")

# shuffled index for user
block_index = list(range(2 * device_num))
random.shuffle(block_index)  # 打乱顺序
print(block_index)
block_size = int(50000 / (2 * device_num))
print(block_size)
shuffled_index = []
# device_num = 1
for i in range(device_num):
    block_start1 = block_index[2 * i] * block_size
    # print("block_start1 = {}".format(block_start1))
    list1 = list(range(block_start1, block_start1 + block_size))
    # print("list1 = {}".format(list1))
    block_start2 = block_index[2 * i + 1] * block_size
    # print("block_start2 = {}".format(block_start2))
    list2 = list(range(block_start2, block_start2 + block_size))
    # print("list2 = {}".format(list2))
    list1.extend(list2)
    # print("listpinjie = {}".format(list1))
    random.shuffle(list1)
    # print("listrandom = {}".format(list1))
    shuffled_index.extend(list1)
# print(shuffled_index)
scipy.io.savemat('shuffled_index.mat', mdict={'shuffled_index': shuffled_index})

# 不同的网络模型对应的不同的输入图片大小
networks_dic = {'alexnet': 252, 'vgg': 224, 'inception': 299, 'resnet': 32, 'densenet': 32, 'mobilenet': 32}
net_choice = 'resnet'  # 改变网络以进行训练
pic_size = networks_dic[net_choice]

if net_choice == "alexnet":  # 加载模型
    myNN = torchvision.models.alexnet(num_classes=10)
elif net_choice == "vgg":
    myNN = torchvision.models.vgg11(num_classes=10)
elif net_choice == "inception":
    myNN = torchvision.models.inception_v3(num_classes=10, aux_logits=False)
elif net_choice == "resnet":
    myNN = ResNet18()
elif net_choice == "densenet":
    myNN = DenseNet121()
elif net_choice == "mobilenet":
    myNN = MobileNetV2()
else:
    pass
myNN = myNN.cuda()
# print(myNN)
# print(type(myNN))
# myNN = myNN.cuda()
myNN = nn.DataParallel(myNN)
# print(myNN)
# print(type(myNN))
# myNN.load_state_dict(torch.load('resnet_model_29epoch.pkl'))


# 选择优化器和损失函数
LR = 0.05
optimizer = torch.optim.SGD(myNN.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# # 初始化操作，进行测试吧
# x_input = torch.zeros([1, 3, pic_size, pic_size], dtype=torch.float).cuda()
# # print(x_input)
# y_output = torch.zeros([1], dtype=torch.long).cuda()
# # print(y_output)
# output = myNN(x_input)  # cnn output
# print(output)
# print(type(output))
# loss = loss_func(output, y_output)  # cross entropy loss
# print(loss)
# print(type(loss))
# optimizer.zero_grad()  # clear gradients for this training step
# loss.backward()  # backpropagation, compute gradients
# optimizer.zero_grad()  # clear gradients for this training step


# 初始化
iteration_index = 1
train_loss_list = []
test_loss_list = []
acc_list = []

print('------------optimization obtain batchsize and compress rate------------------------')

b = [150, 150, 150, 150]
c = [20, 20, 20, 20]

# 加载测试集
data_transform = transforms.Compose([
    # transforms.Resize(pic_size),
    # transforms.RandomCrop(pic_size, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=data_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
# print(testset)
# save model
# myNN.cpu()
file_name = "./server_model_acc/server_model_initial" + ".pkl"
torch.save(myNN.state_dict(), file_name)

print('---------------start device--------------')
# start users
for i in range(1, device_num + 1):
    os.system("bash ./device" + str(i) + "_acc.sh")

# built connect
print('---------------built connect--------------')
conn1 = listener1.accept()
print("connection 1 accepted from:", listener1.last_accepted)
conn2 = listener2.accept()
print("connection 2 accepted from:", listener2.last_accepted)
conn3 = listener3.accept()
print("connection 3 accepted from:", listener3.last_accepted)
# conn4 = listener4.accept()
# print("connection 4 accepted from:", listener4.last_accepted)
# conn5 = listener5.accept()
# print("connection 5 accepted from:", listener5.last_accepted)

print('---------send batchsize compress rate LR and net_choice---------------')
conn1.send(gpu_id)
conn2.send(gpu_id)
conn3.send(gpu_id)
# conn4.send(gpu_id)
# conn5.send(gpu_id)

conn1.send(str(device_num))
conn2.send(str(device_num))
conn3.send(str(device_num))
# conn4.send(str(device_num))
# conn5.send(str(device_num))

conn1.send(net_choice)
conn2.send(net_choice)
conn3.send(net_choice)
# conn4.send(net_choice)
# conn5.send(net_choice)

conn1.send(str(LR))
conn2.send(str(LR))
conn3.send(str(LR))
# conn4.send(str(LR))
# conn5.send(str(LR))

conn1.send(str(b[0]))
conn2.send(str(b[1]))
conn3.send(str(b[2]))
# conn4.send(str(b[3]))
# conn5.send(str(b[4]))

conn1.send(str(c[0]))
conn2.send(str(c[1]))
conn3.send(str(c[2]))
# conn4.send(str(c[3]))
# conn5.send(str(c[4]))

try:
    conn1.recv()
    conn2.recv()
    conn3.recv()
    # conn4.recv()
    # conn5.recv()
except:
    print('connection break! exit!')
print('--------------prepare done, prepare train------------------')

test_loss = 0.0
test_acc = 0.0

while True:  # loop over the dataset multiple times
    # exit condition
    # if os.path.isfile("run_flag.txt"):
    if iteration_index < 3001:  # 在训练1000次之后退出，并保存
        pass
    else:
        file_name = "train_loss_acc" + str(iteration_index) + ".mat"
        scipy.io.savemat(file_name, mdict={'train_loss_acc': train_loss_list})
        file_name = "acc_acc" + str(iteration_index) + ".mat"
        scipy.io.savemat(file_name, mdict={'acc_acc': acc_list})
        file_name = "test_loss_acc" + str(iteration_index) + ".mat"
        scipy.io.savemat(file_name, mdict={'test_loss_acc': test_loss_list})
        conn1.send(str(-1))
        conn2.send(str(-1))
        conn3.send(str(-1))
        # conn4.send(str(-1))
        # conn5.send(str(-1))
        print('Manually Exit!')
        break

    file_name = "./server_model_acc/server_model" + ".pkl"  # broadcast parameter
    torch.save(myNN.state_dict(), file_name)
    grad_list = []  # broadcast gradient
    for index, (name, param) in enumerate(myNN.named_parameters()):
        grad_list.append(param.grad)
    torch.save(grad_list, './grads_acc/server_grads' + '.pkl')

    conn1.send('synchronization')
    conn2.send('synchronization')
    conn3.send('synchronization')
    # conn4.send('synchronization')
    # conn5.send('synchronization')
    print('--------download grad synchronization waiting!------------')
    try:
        conn1.recv()
        conn2.recv()
        conn3.recv()
        # conn4.recv()
        # conn5.recv()
    except:
        print('connection break! exit!')
        break
    print('------------------complete download,start compute gradient locally------------------------')
    # FOR TEST
    try:
        conn1.recv()
        conn2.recv()
        conn3.recv()
        # conn4.recv()
        # conn5.recv()

    except:
        print('aaaa break! exit!')
        break

    print('----------collect gradient & loss and average------------------')
    optimizer.zero_grad()  # clear gradients for this training step
    grads = []
    for i in range(1, device_num + 1):
        grads.append(torch.load('./grads_acc/grads' + str(i) + '.pkl'))
    for index, (name, param) in enumerate(myNN.named_parameters()):
        param.grad = grads[0][index] * b[0]
        for i in range(1, device_num):
            param.grad = param.grad + grads[i][index] * b[i]
        param.grad = param.grad / sum(b)

    # save loss and delay
    losses = []
    for i in range(1, device_num + 1):
        losses.append(torch.load('./loss_acc/loss' + str(i) + '.pkl'))
    loss = losses[0].item() * b[0]
    for i in range(1, device_num):
        loss = loss + losses[i].item() * b[i]
    loss = loss / sum(b)
    print('iteration', iteration_index, '  loss: ', loss, '  test loss: ', test_loss, '  test acc:  ', test_acc)
    # save loss and delay
    train_loss_list.append(loss)

    # -------gradient_update--------
    optimizer.step()  # apply gradients
    # 每隔20步，保存一次带有训练次数的模型，以及带有训练次数的损失函数，以及统计测试集的正确率和损失函数
    if iteration_index % 20 == 0:
        # save acc and loss
        correct = 0
        total = 0

        test_index = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = myNN(images)
                loss = loss_func(outputs, labels)
                test_loss = test_loss + loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_index = test_index + 1
        test_loss = test_loss.item() / test_index
        # print('Loss of the network on the 10000 test images: ', test_loss)
        test_loss_list.append(test_loss)
        test_acc = correct / total
        # print('Accuracy of the network on the 10000 test images:', test_acc)
        acc_list.append(test_acc)
        file_name = "train_loss_acc_tmp" + ".mat"
        scipy.io.savemat(file_name, mdict={'train_loss_acc_tmp': train_loss_list})
        file_name = "acc_list_acc_tmp" + ".mat"
        scipy.io.savemat(file_name, mdict={'acc_list_acc_tmp': acc_list})
        file_name = "test_loss_acc_tmp" + ".mat"
        scipy.io.savemat(file_name, mdict={'test_loss_acc_tmp': test_loss_list})
    iteration_index = iteration_index + 1
print('Finished Training')
# conn1.send(str(-1))
# conn2.send(str(-1))
# conn3.send(str(-1))
# conn4.send(str(-1))
# conn5.send(str(-1))
# print('Manually Exit!')

conn1.close()
conn2.close()
conn3.close()
# conn4.close()
# conn5.close()
listener1.close()
listener2.close()
listener3.close()
# listener4.close()
# listener5.close()
