import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class word2vec(nn.Module):
    def __init__(self,vocab_size,context_size,embedding_size):
        super(word2vec,self).__init__()
        self.fc1 = nn.Linear(vocab_size,embedding_size)
        self.fc2 = nn.Linear(embedding_size,context_size*vocab_size)
        self.context_size = context_size
        self.vocab_size = vocab_size
    def forward(self,x):
        x = self.fc1(x)
        embedding = x
        x = self.fc2(embedding)
        return [F.softmax(x[:,i:i+self.vocab_size]) for i in xrange(0,self.context_size*self.vocab_size,self.vocab_size)]+[embedding]


def make_corpa(data):
    vocab = ""
    for i in data:
        vocab = vocab + " " + i
    vocab.lstrip(" ")
    corpa = {}
    all_words = vocab.split(" ")
    for i in all_words:
        if i not in corpa.keys():
            corpa[i] = len(corpa)
    
    return [corpa,len(corpa),corpa.keys()]

def conv_vect(word,corpa):
    temp = torch.FloatTensor(1,len(corpa)).zero_()
    temp[0][corpa[word]] = 1
    return temp

def count_dict(data):
    counter = dict()
    for i in data:
        for j in i.split(" "):
            if j in counter.keys():
                counter[j] = counter[j] + 1
            else:
                counter[j] = 1
    
    return counter

def subsample(context,word,threshold,data):
    counter = count_dict(data)
    new_context = [] 
    new_word = []
    total_count  = 0
    #print counter
    for i in context:
        if 1 - (threshold/counter[i])**(0.5) > 0.5:
            new_context.append(i)
    
    if 1 - (threshold/counter[i])**(0.5) > 0.5:
        new_word.append(word)
    
    return new_context,new_word

def train_word2vec(embedding_dim,number_of_epochs,data,context_size,threshold,discount):
    corpa = make_corpa(data)[0]
    vocab_size = make_corpa(data)[1]
    model = word2vec(vocab_size,context_size,embedding_dim)
    loss = nn.CrossEntropyLoss()
    context,word = make_training_data(data,context_size/2)
    new_context = []
    new_word = []
    for i in xrange(len(context)):
        temp_context,temp_word = subsample(context[i],word[i],threshold,data)
        if temp_word:
            new_context.append(temp_context)
            new_word.append(temp_word[0])
    
    counter,corpa = phrase(data,discount)
    print corpa
    optimizer = optim.SGD(model.parameters(),lr= 0.01)
    for epoch in xrange(number_of_epochs):
        for i in xrange(len(new_context)):
            l = Variable(torch.FloatTensor(1).zero_(),requires_grad = True)
            context_vec_tmp = [Variable(conv_vect(j,corpa)).max(1)[1][0] for j in new_context[i]]
            context_vec = torch.cat(context_vec_tmp)
            word_vec = Variable(conv_vect(new_word[i],corpa))
            predict = model(word_vec)
            for k in xrange(context_size):
                l = l + loss(predict[k],context_vec[k])
            
            model.zero_grad()
            l.backward()
            optimizer.step()
            
            
    
    return model

def find(string1,string2):
    if string1.find(string2):
        return 1
    else:
        return 0

def phrase(data,discount):
    count = count_dict(data)
    corpus = make_corpa(data)[0]
    bi_gram = dict()
    for i in data:
        temp = i.split(" ")
        for j in xrange(len(temp)-1):
            bi_gram.append(temp[j]+" "+temp[j+1])
    
    score = dict()
    for i in bi_gram:
        score[i] = 0
        
    for i in bi_gram:
        for j in data:
            score[i] = score[i] + find(i,j)
    
    
    for i in score.keys():
        if (float(score[i]) - float(discount))/(float(score[i.split(" ")[0]])*float(score[i.split(" ")[1]])) > 0.5:
            count[i] = score[i]
            corpus[i] = len(corpus)
            
    return count,corpus

def make_training_data(data,context_size):
    context = []
    word = []
    for i in data:
        temp = i.split(" ")
        for j in xrange(context_size,len(temp)-context_size,1):
            context.append([temp[j - context_size],temp[j - context_size + 1],temp[j - context_size + 2],temp[j + context_size - 2],temp[j + context_size - 1],temp[j + context_size]])
            word.append(temp[j])
    
    return context,word