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
                       transforms.Normalize((0.1307,), (0.3081,)),                       
                   ])),
batch_size= 1,shuffle=True)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(100,1024*7*7)
        self.conv1 = nn.ConvTranspose2d(1024,512,kernel_size=(2,2),stride = 2)
        self.conv2 = nn.ConvTranspose2d(512,256,kernel_size = (2,2),stride = 2)
        self.conv3 = nn.ConvTranspose2d(256,1,kernel_size=(1,1),stride = 1)
        self.fc2 = nn.Linear(28*28,2)
    
    def forward(self,x):
        x = self.fc1(x).view(1,1024,7,7)
        x = self.conv1(x)
        print x.size()
        x = self.conv2(x)
        print x.size()
        x1 = self.conv3(x)
        print x1.size()
        x2 = F.softmax(self.fc2(x1.view(1,28*28)))
        return x2,x1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=(2,2),stride = 2)
        self.conv2 = nn.Conv2d(128,256,kernel_size=(2,2),stride = 2)
        self.conv3 = nn.Conv2d(256,512,kernel_size=(2,2),stride = 2)
        self.conv4 = nn.Conv2d(512,1024,kernel_size=(2,2),stride = 2)
        self.fc1 = nn.Linear(1024,2)
        self.change = nn.UpsamplingNearest2d((1,1,64,64))
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x.view(1,1024))
        x = F.softmax(x)
        return x

def train(num_epochs,g_steps,d_steps):
    gen = Generator()
    disc = Discriminator()
    d_real_loss = nn.CrossEntropyLoss()
    d_fake_loss = nn.CrossEntropyLoss()
    g_loss = nn.CrossEntropyLoss()
    d_optimizer = optim.SGD(disc.parameters(),lr = 0.01)
    g_optimizer = optim.SGD(gen.parameters(),lr  = 0.01)
    check_d_steps = 0
    check_g_steps = 0
    flag_d = 0
    flag_g = 0
    for i in xrange(num_epochs):
        for batch_idx,(data,target) in enumerate(train_loader):
            d = disc(Variable(data))
            d_l = d_real_loss(d,Variable(torch.LongTensor([1])))
            disc.zero_grad()
            d_l.backward()
            d_optimizer.step()
                
            d = disc(gen(Variable(torch.randn(1,100)))[1].detach())
            d_l = d_fake_loss(d,Variable(torch.LongTensor([0])))
            disc.zero_grad()
            print d_l
            d_l.backward()
            d_optimizer.step()
            if check_d_steps < d_steps:
                check_d_steps = check_d_steps + 1
            else:
                for _ in xrange(g_steps):
                    g = gen(Variable(torch.randn(1,100)))[0]
                    g_l = g_loss(g,Variable(torch.LongTensor([1])))
                    gen.zero_grad()
                    print g_l
                    g_l.backward()
                    g_optimizer.step()
    
    return gen



