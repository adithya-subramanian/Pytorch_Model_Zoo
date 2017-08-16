import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
vocab_size = 0
num_labels = 3
corpa = {}
label_corpa = {'good':0,'bad':1,'neutral':2}
class BOWClassifier(nn.Module):
    def __init__(self,vocab_size,num_labels):
        super(BOWClassifier,self).__init__()
        self.fc1 = nn.Linear(vocab_size,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,250)
        self.fc4 = nn.Linear(250,125)
        self.fc5 = nn.Linear(125,50)
        self.fc6 = nn.Linear(50,num_labels)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.softmax(self.fc6(x))
        return x

def make_corpa(data):
    corpa = {}
    for i in xrange(len(data)):
        for word in data[i].split(" "):
            if word not in corpa.keys():
                corpa[word] = len(corpa)
    
    num_labels = 3
    vocab_size = len(corpa)
    return corpa
        
def make_vect(sentence,corpa):
    z = torch.FloatTensor(1,len(corpa)).zero_()
    for word in sentence.split(" "):
        z[0][corpa[word]] += 1
    return z

def make_target(labels,label_corpa):
    print labels
    return torch.LongTensor([label_corpa[labels]])

def run(num_of_epochs,data,labels,num_labels):
    corpa = make_corpa(data)
    model = BOWClassifier(len(corpa),num_labels)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in xrange(num_of_epochs):
        for i in xrange(len(data)):
            bow_vec = Variable(make_vect(data[i],corpa))
            target = Variable(make_target(labels[i],label_corpa))
            prob = model(bow_vec)
            loss = loss_function(prob,target)
            model.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def test(model,test_data,labels):
    for i in test_data:
        bow_vec = Variable(make_vect(i,corpa))
        return model(bow_vec)
