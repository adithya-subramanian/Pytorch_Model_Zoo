import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,n_layers,hidden_size):
        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,n_layers,dropout = 0.5)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
    def init_hidden_cell(self):
        hidden = (Variable(torch.randn(1,1,self.hidden_size)),Variable(torch.randn(1,1,self.hidden_size)))
        return hidden
    
    def forward(self,x):
        vect = []
        for i in xrange(len(x)):
            vect.append(self.embedding(x[i].max(1)[1]))
        
        hidden = self.init_hidden_cell()
        output,hidden = self.lstm(torch.cat(vect),hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,input_size,n_layers):
        super(Decoder,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,dropout = 0.5)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.fc1 = nn.Linear(hidden_size,vocab_size)
    def forward(self,hidden):
        output,hidden = self.lstm(Variable(torch.zeros(1,1,self.input_size)),hidden)
        return F.softmax(self.fc1(hidden[0].view(-1,self.hidden_size))),hidden
        
def make_corpus(data):
    corpa = {"#":0}
    for i in data:
        for j in i.split(" "):
            if j not in corpa.keys():
                corpa[j] = len(corpa)
    return corpa

def make_vect(word,corpa):
    temp = torch.FloatTensor(1,len(corpa)).zero_()
    temp[0][corpa[word]] = 1.0
    return temp

def train(data1,data2,embedding_size,n_layers,input_size,hidden_size,num_epochs):
    corpa_lang1 = make_corpus(data1)
    corpa_lang2 = make_corpus(data2)
    #print corpa_lang1
    enc = Encoder(len(corpa_lang1),embedding_size,n_layers,hidden_size)
    dec = Decoder(len(corpa_lang2),hidden_size,input_size,n_layers)
    loss = nn.CrossEntropyLoss()
    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = optim.SGD(params,lr= 0.01)
    for i in xrange(num_epochs):
        for j in xrange(len(data1)):
            l = 0
            ip_vec = [Variable(make_vect(k,corpa_lang1),requires_grad= True) for k in data1[j].split(" ")]
            ip_vec = ip_vec + [Variable(make_vect("#",corpa_lang1),requires_grad = True)]
            op1,op2 = dec(enc(ip_vec))
            for m in xrange(len(data2[j].split(" "))+1):
                if m == len(data2[j].split(" ")):
                    op_vec = Variable(torch.FloatTensor([corpa_lang2["#"]]))
                    op_vec.data = torch.Tensor.long(op_vec.data)
                    op1,op2 = dec(op2)
                    l = l + loss(op1,op_vec)
                else:
                    op_vec = Variable(torch.FloatTensor([corpa_lang2[data2[j].split(" ")[m]]]))
                    op_vec.data = torch.Tensor.long(op_vec.data)
                    if m == 0:
                        l=l+loss(op1,op_vec)
                    else:
                        op1,op2 = dec(op2)
                        l = l + loss(op1,op_vec)
                
            
            enc.zero_grad()
            dec.zero_grad()
            print l
            l.backward(retain_variables = True)
            optimizer.step()
    
    return enc,dec