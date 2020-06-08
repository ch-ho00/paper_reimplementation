import warnings
warnings.filterwarnings("ignore")
import random
import torchvision.transforms as transforms
import torch.optim as optim
import os,pickle
import torch.nn as nn
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import scipy.misc
from utils import *
import time
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import glob
import cv2 as cv
from affine import Affine
import matplotlib.pyplot as plt

def get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = y_train
    Y_test = y_test

    return (X_train, Y_train), (X_test, Y_test)


def get_gen(dataset,set_name, batch_size, t=0, s=1,
            shuffle=True,n=10000):
    '''
    Create image generator with images scaled by 1/s and translated by t
    Input:
        dataset = name of dataset
        set_name = train/test
        scale = single float number 
    Output:
        ImageDataGenerator
    '''
    if dataset == "MNIST":
        if set_name == 'train':
            (X, Y), _ = get_mnist_dataset()
        elif set_name == 'test':
            _, (X, Y) = get_mnist_dataset()

        X = X[:n]
        Y = Y[:n]
        image_gen = ImageDataGenerator(
            zoom_range=(s-0.001,s+0.001),
            width_shift_range=t,
            height_shift_range=t
        )

        gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle, )
    elif dataset == "STL":
        if set_name == 'train':
            (X, Y), _ = get_stl_dataset()
        elif set_name == 'test':
            _, (X, Y) = get_stl_dataset()
        image_gen = ImageDataGenerator(
            zoom_range=s,
            width_shift_range=t,
            height_shift_range=t
        )
        gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen

def get_gen_rand(dataset,set_name, batch_size, t, s,
            shuffle=False, n =10000):
    '''
    Create image generator with images scaled by 1/s and translated by t
    Input:
        dataset = name of dataset
        set_name = train/test
        scale = single float number 
    Output:
        ImageDataGenerator
    '''
    if dataset == "MNIST":
        if set_name == 'train':
            (X, Y), _ = get_mnist_dataset()
        elif set_name == 'test':
            _, (X, Y) = get_mnist_dataset()

        # Y2 = np.zeros((Y.size, 10))
        # Y2[np.arange(Y.size),Y] = 1
        image_gen = ImageDataGenerator(
            zoom_range=(0.3,0.7),
            width_shift_range=t,
            height_shift_range=t
        )
        gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    elif dataset == "STL":
        if set_name == 'train':
            (X, Y), _ = get_stl_dataset()
        elif set_name == 'test':
            _, (X, Y) = get_stl_dataset()
        image_gen = ImageDataGenerator(
            zoom_range=s,
            width_shift_range=t,
            height_shift_range=t
        )
        gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen
def aeae_loss(inputs, x_pred, trans_x, trans_x_pred, trans_x_pred_2):
    sum_ = 0

    m1 = inputs - x_pred
    m2 = trans_x - trans_x_pred
    m3 = trans_x - trans_x_pred_2
    print("Reconstruction cost",sum([torch.norm(m1[i]) for i in range(m1.size(0))]))
    sum_ += sum([torch.norm(m1[i]) for i in range(m1.size(0))])
    sum_ += sum([torch.norm(m2[i]) for i in range(m2.size(0))])
    sum_ += sum([torch.norm(m3[i]) for i in range(m3.size(0))])
    return sum_

def train_network(net,trainloader,num_batches,sigma,e):

    init_rate = 0.001
    weight_decay = 0.04
    step_size = 10
    gamma = 0.7
    
    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(),lr=init_rate, momentum=0.9,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    net = net.train()

    torch.cuda.empty_cache()
    scheduler.step()
    loss_sum = 0
    for i in range(num_batches):
        inputs, _ = next(trainloader)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.permute(0,3,1,2)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        trans_x, x_pred, trans_x_pred,trans_x_pred_2 , original_embedding, trans_embedding, trans_embedding_2 = net(inputs,sigma)
        loss = aeae_loss(inputs.squeeze(1), x_pred.squeeze(1), trans_x, trans_x_pred.squeeze(1), trans_x_pred_2.squeeze(1))
        loss_sum += loss
        loss.backward()
        optimizer.step()
    print("\tTraining Loss =",loss_sum) # //(num_batches* inputs.size(0))
    if e % 10 == 0:
        display_rand_output(inputs, trans_x,x_pred, trans_x_pred,trans_x_pred_2,index =[10,100],epoch=e)
    net = net.eval()
    return net

def display_rand_output(x, trans_x, x_pred, trans_x_pred,trans_x_pred_2, epoch, n=5, index=None):
    '''
    In order the check the effectiveness of the model, display random samples of output and compare with ground truth.
    Input:
        x = original image || torch.Tensor (B,C,H,W)
        trans_x = transformed(i.e. scaled) image || torch.Tensor (B,C,H,W)
        x_pred = predicted image through path 1 || torch.Tensor (B,C,H,W)
        trans_x_pred =  predicted image through path 2 || torch.Tensor (B,C,H,W)
        trans_x_pred_2 = predicted image through path 3 || torch.Tensor (B,C,H,W)
        n = number of samples to take
        index = particular list of image index to track 
    Output:
        Image file in ./result/        
    '''
    x = x.cpu().squeeze(1).detach()
    trans_x = trans_x.cpu().squeeze(1).detach()
    trans_x_pred = trans_x_pred.cpu().squeeze(1).detach()
    trans_x_pred_2 = trans_x_pred_2.cpu().squeeze(1).detach()
    x_pred = x_pred.cpu().squeeze(1).detach()


    if index == None:
        for i in range(5):
            rand_int = random.randint(0,x.size(0)-1)
            fig = plt.figure()
            a = fig.add_subplot(2,3,1)
            plt.imshow(x.numpy()[rand_int])
            a.set_title('Original Image', fontsize= 10)
                    
            a = fig.add_subplot(2,3,2)
            plt.imshow(x_pred.numpy()[rand_int])
            a.set_title('Predicted Original Image',fontsize= 10)

            a = fig.add_subplot(2,3,4)
            plt.imshow(trans_x.numpy()[rand_int])
            a.set_title('Transformed Image',fontsize= 10)

            a = fig.add_subplot(2,3,5)
            plt.imshow(trans_x_pred.numpy()[rand_int])
            a.set_title('Predicted Transformed Image 1',fontsize= 10)

            a = fig.add_subplot(2,3,6)
            plt.imshow(trans_x_pred_2.numpy()[rand_int])
            a.set_title('Predicted Transformed Image 2',fontsize= 10)

            plt.savefig('./result/result_'+str(i)+'_'+str(epoch)+'.PNG')
            plt.close()
    else:
        for i in index:
            rand_int = random.randint(0,x.size(0)-1)
            fig = plt.figure()
            a = fig.add_subplot(2,3,1)
            plt.imshow(x.numpy()[i])
            a.set_title('Original Image',fontsize= 10)
                    
            a = fig.add_subplot(2,3,2)
            plt.imshow(x_pred.numpy()[i])
            a.set_title('Predicted Original Image',fontsize= 10)

            a = fig.add_subplot(2,3,4)
            plt.imshow(trans_x.numpy()[i])
            a.set_title('Transformed Image',fontsize= 10)

            a = fig.add_subplot(2,3,5)
            plt.imshow(trans_x_pred.numpy()[i])
            a.set_title('Predicted Transformed Image 1',fontsize= 10)

            a = fig.add_subplot(2,3,6)
            plt.imshow(trans_x_pred_2.numpy()[i])
            a.set_title('Predicted Transformed Image 2',fontsize= 10)

            plt.savefig('./result/result_'+str(i)+'_'+str(epoch)+'.PNG')
    # print("Sample test results saved")



def test_network(net,testloader,total,num_batches,sigma):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.eval()
    correct = torch.tensor(0)
    
    for i in range(num_batches):
        images, labels = next(testloader)
        
        images, labels = torch.from_numpy(images), torch.from_numpy(labels)
        images = images.permute(0,3,1,2)
        images = images.to(device)
        labels = labels.to(device)

        outputs, embedding = net(images,sigma)
        
        _, predicted = torch.max(outputs, 1)
        correct = correct + torch.sum(predicted == labels)
        torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy, embedding

