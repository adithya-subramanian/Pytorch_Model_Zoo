import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CBOW(nn.Module):
    def __init__(self,vocab_size,embedding_size,context_size):
        super(CBOW,self).__init__()
        self.fc1 = nn.Linear(vocab_size,embedding_size)
        self.fc2 = nn.Linear(embedding_size,vocab_size)
    
    def forward(self,x):
        y = []
        for i in xrange(0,174,29):
            y.append(self.fc1(x[:,i:i+29]))
        
        embedding = Variable(torch.zeros(1,128))
        for i in xrange(len(y)):
            embedding = embedding + y[i]
            
        embedding = embedding/len(y)
        x = self.fc2(embedding)
        return [F.softmax(x),embedding]

def make_corpa(data):
    vocab = ""
    for i in data:
        vocab = vocab + " " + i
    vocab.strip(" ")
    corpa = {}
    all_words = list(set(vocab.split(" ")))
    for i in xrange(len(all_words)):
        corpa[all_words[i]] = i
    
    return [corpa,len(corpa),corpa.keys()]

def conv_vect(word,corpa):
    temp = torch.FloatTensor(1,len(corpa)).zero_()
    temp[0][corpa[word]] = 1.0
    return temp

def train_word2vec(vocab_size,embedding_dim,number_of_epochs,data):
    model = CBOW(vocab_size,embedding_dim,6)
    loss = nn.CrossEntropyLoss()
    context,word = make_training_data(data,3)
    corpa = make_corpa(data)[0]
    optimizer = optim.SGD(model.parameters(),lr= 0.01)
    for epoch in xrange(number_of_epochs):
        for i in xrange(len(context)):
            context_vec_tmp = [conv_vect(j,corpa) for j in context[i]]
            context_vec = Variable(torch.cat(tuple([context_vec_tmp[j] for j in xrange(len(context_vec_tmp))]),1),requires_grad = True)
            word_vec = Variable(conv_vect(word[i],corpa),requires_grad = True)
            predict = model(context_vec)[0]
            model.zero_grad()
            l = loss(predict,word_vec.max(1)[1][0])
            print l.creator.previous_functions[0][0].previous_functions[0][0]
            l.backward()
            optimizer.step()
    
    return model

def make_training_data(data,context_size):
    context = []
    word = []
    for i in data:
        temp = i.split(" ")
        for j in xrange(context_size,len(temp)-context_size,1):
            context.append([temp[j - context_size],temp[j - context_size + 1],temp[j - context_size + 2],temp[j + context_size - 2],temp[j + context_size - 1],temp[j + context_size]])
            word.append(temp[j])
    
    return context,word



