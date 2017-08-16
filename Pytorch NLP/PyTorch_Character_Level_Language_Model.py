import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class next_char(nn.Module):
    def __init__(self,number_of_characters,input_size):
        super(next_char,self).__init__()
        self.lstm = nn.LSTM(input_size,input_size)
        self.number_of_characters = number_of_characters
        self.input_size = input_size
    def inithiddenlayer(self):
        return (Variable(torch.rand(1,1,self.input_size),requires_grad = True),Variable(torch.rand(1,1,self.input_size),requires_grad = True))
    
    def forward(self,x):
        out,hidden = self.lstm(torch.cat(x).view(-1,1,self.input_size),self.inithiddenlayer())
        i = -1
        out,hidden = self.lstm(Variable(torch.FloatTensor(1,1,self.input_size).zero_(),requires_grad = True),hidden)
        op = nn.Softmax()(hidden[0][0])
        flag = 0
        while i == -1 or (flag == 0 and i < 10):
            out,hidden = self.lstm(Variable(torch.FloatTensor(1,1,self.input_size).zero_(),requires_grad = True),hidden)
            #print nn.Softmax()(hidden[0][0]).max(1)[1].data[0][0]
            if nn.Softmax()(hidden[0][0]).max(1)[1].data[0][0] == 0:
                flag = 1

            op = torch.cat([op,nn.Softmax()(hidden[0][0])])
            i = i + 1
        
        return op


def make_vect(letter,corpa):
    vect = torch.FloatTensor(1,len(corpa)).zero_()
    vect[0][corpa[letter]] = 1.0
    return vect

def make_corpa(data):
    corpa = {"#" : 0}
    for i in data.split(" "):
        for j in list(i.lower()):
            if j not in corpa.keys():
                corpa[j] = len(corpa)
    
    return corpa

def make_training_data(data,number_of_characters):
    corpa = make_corpa(data)
    all_words = data.split(" ")
    
    input_data = []
    output_data = []
    for i in all_words:
        temp1 = []
        temp2 = []
        i = i.lower()
        for j in xrange(len(i)):
            if j < number_of_characters or j == number_of_characters:
                temp1.append(Variable(make_vect(i[j].lower(),corpa),requires_grad = True))
            else:
                output_data.append(Variable(torch.LongTensor([make_vect(i[j].lower(),corpa).max(1)[1][0][0]])))
            
        output_data.append(Variable(torch.LongTensor([make_vect("#",corpa).max(1)[1][0][0]])))
        input_data.append(temp1)
        
    return input_data,torch.cat(output_data),corpa

def train_the_network(data,number_of_characters):
    input_data,output_data,corpa = make_training_data(data,number_of_characters)
    model = next_char(number_of_characters,len(corpa))
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr= 0.01)
    k = 0
    for i in xrange(len(input_data)):
        op = model(input_data[i])
        l = Variable(torch.FloatTensor(1).zero_(),requires_grad = True)
        j = 0
        while output_data.data[k] != 0:
            print op[j]
            #print output_data[k]
            l = l + loss(op[j].view(1,-1),output_data[k])
            k = k + 1
            j = j + 1
        
        model.zero_grad()
        l.creator.previous_functions[0][0].previous_functions[0][0]
        l.backward()
        optimizer.step()
    return model