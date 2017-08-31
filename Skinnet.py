import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
import torchvision.models as models



def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch/lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


class VGGNET(nn.Module):
    def __init__(self):
        super(VGGNET,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,stride=1)
        self.conv2 = nn.Conv2d(64,64,3,stride=1)
        self.conv3 = nn.Conv2d(64,128,3,stride=1)
        self.conv4 = nn.Conv2d(128,128,3,stride=1)
        self.conv5 = nn.Conv2d(128,256,3,stride=1)
        self.conv6 = nn.Conv2d(256,256,3,stride=1)
        self.conv7 = nn.Conv2d(256,256,3,stride=1)
        self.conv8 = nn.Conv2d(256,512,3,stride=1)
        self.conv9 = nn.Conv2d(512,512,3,stride=1)
        self.conv10 = nn.Conv2d(512,512,3,stride=1)
        self.conv11 = nn.Conv2d(512,512,3,stride=1)
        self.conv12 = nn.Conv2d(512,512,3,stride=1)
        self.conv13 = nn.Conv2d(512,512,3,stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.bn11 = nn.BatchNorm2d(512)
        self.bn12 = nn.BatchNorm2d(512)
        self.bn13 = nn.BatchNorm2d(512)
        self.maxpool1 = nn.MaxPool2d(2,stride = 2)
        self.maxpool2 = nn.MaxPool2d(2,stride=2)
        self.maxpool3 = nn.MaxPool2d(2,stride=2)
        self.maxpool4 = nn.MaxPool2d(2,stride=2)
        self.maxpool5 = nn.MaxPool2d(2,stride=2)
        #self.fc0 = nn.Linear(1024,512)
        self.fc1 = nn.Linear(512,4096)
        self.fc2 = nn.Linear(4096,1000)
        self.fc3 = nn.Linear(1000,2)
    
    def forward(self,x):
        #print x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = x = self.bn2(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.maxpool3(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.maxpool4(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = F.dropout(x)
        x = F.elu(x)
        x = self.maxpool5(x)
        x = self.fc1(x.view(1,-1))
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x

net = VGGNET().cuda()

# for plotting kernels
# mm = net.double()
# filters = mm.modules
# body_model = [i for i in mm.children()][0]
# layer1 = body_model
# tensor = layer1.weight.data.numpy()
# plot_kernels(params[-2].data.numpy())

skin_dataset = datasets.ImageFolder(root='../train_dat')

dataset_loader = torch.utils.data.DataLoader(skin_dataset,
                                             batch_size=1, shuffle=True,
                                             num_workers=4)

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch/lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

loss = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(net.parameters(),lr= 0.001)
transform1 = transforms.Scale(224)
transform2 = transforms.CenterCrop(224)
transform3 = transforms.RandomCrop(224,padding = 4)
transform4 = transforms.RandomHorizontalFlip()
transform6 = transforms.ToTensor()
for epoch in xrange(100):
    print epoch
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[j][0]))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[j][1]]).cuda()))
        net.zero_grad()
        l.backward()
        total_loss = l
        exp_lr_scheduler(optimizer,epoch).step()
    
    if epoch%10 == 0:
        print total_loss

for epoch in xrange(100):
    print epoch
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[0][0].rotate(random.randint(1,360))))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        total_loss = l
        exp_lr_scheduler(optimizer,epoch).step()
    
    if epoch%10 == 0:
        print total_loss

for epoch in xrange(100):
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform3(transform1(dataset_loader.dataset[0][0]))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        print l
        exp_lr_scheduler(optimizer,epoch).step()

for epoch in xrange(100):
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform4(transform1(dataset_loader.dataset[0][0])))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        print l
        exp_lr_scheduler(optimizer,epoch).step()

for epoch in xrange(100):
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[0][0]))).view(1,3,224,-1).cuda() + 0.0011*torch.randn(1,3,224,224).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        print l
        exp_lr_scheduler(optimizer,epoch).step()

torch.save(net,"../SkinLesNet.dat")

accuracy = 0
for j in xrange(len(dataset_loader.dataset.imgs)):
    if torch.max(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[j][0])))).view(1,3,224,-1).cuda()),1)[1].data[0][0] == dataset_loader.dataset[j][1]:
        print dataset_loader.dataset[j][1]
        accuracy = accuracy + 1
    
print accuracy

net = models.resnet18(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

net.fc = nn.Linear(512,2)
net = net.cuda()

for epoch in xrange(100):
    print epoch
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[j][0]))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[j][1]]).cuda()))
        net.zero_grad()
        l.backward()
        total_loss = l
        exp_lr_scheduler(optimizer,epoch).step()
    
    if epoch%10 == 0:
        print total_loss

for epoch in xrange(100):
    print epoch
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[0][0].rotate(random.randint(1,360))))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        total_loss = l
        exp_lr_scheduler(optimizer,epoch).step()
    
    if epoch%10 == 0:
        print total_loss

for epoch in xrange(100):
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform3(transform1(dataset_loader.dataset[0][0]))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        print l
        exp_lr_scheduler(optimizer,epoch).step()

for epoch in xrange(100):
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform4(transform1(dataset_loader.dataset[0][0])))).view(1,3,224,-1).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        print l
        exp_lr_scheduler(optimizer,epoch).step()

for epoch in xrange(100):
    total_loss = 0
    for j in xrange(int(0.75*float(len(dataset_loader.dataset.imgs)))):
        l = loss(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[0][0]))).view(1,3,224,-1).cuda() + 0.0011*torch.randn(1,3,224,224).cuda(),requires_grad = True)),Variable(torch.LongTensor([dataset_loader.dataset[0][1]]).cuda()))
        net.zero_grad()
        l.backward()
        print l
        exp_lr_scheduler(optimizer,epoch).step()

accuracy = 0
for j in xrange(len(dataset_loader.dataset.imgs)):
    if torch.max(net(Variable(transform6(transform2(transform1(dataset_loader.dataset[j][0])))).view(1,3,224,-1).cuda()),1)[1].data[0][0] == dataset_loader.dataset[j][1]:
        print dataset_loader.dataset[j][1]
        accuracy = accuracy + 1

print accuracy