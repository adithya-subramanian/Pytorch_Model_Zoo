import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils import data
from torchvision import datasets,transforms
train_loader = data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.0,), (0.02,)),                       
                   ])),
batch_size= 128,shuffle=True)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(100,1024*7*7)
        self.conv1 = nn.ConvTranspose2d(1024,512,kernel_size=(2,2),stride = 2)
        self.conv2 = nn.ConvTranspose2d(512,256,kernel_size = (2,2),stride = 2)
        self.conv3 = nn.ConvTranspose2d(256,1,kernel_size=(1,1),stride = 1)
        self.fc2 = nn.Linear(28*28,2)
        self.batchnorm1 = nn.BatchNorm2d(512,affine=False)
        self.batchnorm2 = nn.BatchNorm2d(256,affine=False)
    
    def forward(self,x):
        x = self.fc1(x).view(128,1024,7,7)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = F.tanh(self.fc2(x1.view(128,28*28)))
        return x2,x1
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=(2,2),stride = 2)
        self.batchnorm1 = nn.BatchNorm2d(128,affine=False)
        self.conv2 = nn.Conv2d(128,256,kernel_size=(2,2),stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(256,affine=False)
        self.conv3 = nn.Conv2d(256,512,kernel_size=(2,2),stride = 2)
        self.batchnorm3 = nn.BatchNorm2d(512,affine=False)
        self.conv4 = nn.Conv2d(512,1024,kernel_size=(2,2),stride = 2)
        self.batchnorm4 = nn.BatchNorm2d(1024,affine=False)
        self.fc1 = nn.Linear(1024,2)
        self.change = nn.UpsamplingNearest2d((128,1,64,64))
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act(x)
        x = self.fc1(x.view(128,1024))
        x = self.act(x)
        return x

def train(num_epochs,g_steps,d_steps):
    gen = Generator()
    disc = Discriminator()
    d_real_loss = nn.CrossEntropyLoss()
    d_fake_loss = nn.CrossEntropyLoss()
    g_loss = nn.CrossEntropyLoss()
    d_optimizer = optim.Adam(disc.parameters(),lr = 0.002,betas = (0.5,0.5))
    g_optimizer = optim.Adam(gen.parameters(),lr  = 0.002,betas = (0.5,0.5))
    check_d_steps = 0
    check_g_steps = 0
    flag_d = 0
    flag_g = 0
    for i in xrange(num_epochs):
        for batch_idx,(data,target) in enumerate(train_loader):
            if data.size()[0] != 96:
                d = disc(Variable(data))
                d_l = d_real_loss(d,Variable(torch.ones(128).long()))
                disc.zero_grad()
                d_l.backward()
                d_optimizer.step()
                
                d = disc(gen(Variable(torch.randn(128,100)))[1].detach())
                d_l = d_fake_loss(d,Variable(torch.zeros(128).long()))
                disc.zero_grad()
                print d_l
                print "discriminator"
                d_l.backward()
                d_optimizer.step()
                if check_d_steps < d_steps:
                    check_d_steps = check_d_steps + 1
                else:
                    for _ in xrange(g_steps):
                        g = gen(Variable(torch.randn(128,100)))[0]
                        g_l = g_loss(g,Variable(torch.ones(128).long()))
                        gen.zero_grad()
                        print g_l
                        print "generator"
                        g_l.backward()
                        g_optimizer.step()
            else:
                continue
    
    return gen

mod = train(1000,3,1)
print mod(Variable(torch.randn(128,100)))