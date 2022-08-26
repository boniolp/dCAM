from torch.nn import functional as F
from torch import topk
from torch import nn
from torch.utils import data
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
from typing import cast, Union, List


class LSTMClassifier(nn.Module):
    def __init__(self, seq_dim, input_dim, hidden_dim, layer_dim, num_classes=10,device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.batch_size = None
        self.hidden = None
        self.input_dim = input_dim
        self.seq_dim = seq_dim
        self.device = device
    
    def forward(self, x):
        #x = torch.reshape(x,(self.input_dim,self.seq_dim))
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if self.device == 'cuda':
            return [t.cuda() for t in (h0, c0)]
        else:
            return [t for t in (h0, c0)]

class RNNClassifier(nn.Module):
    def __init__(self, seq_dim, input_dim, hidden_dim, layer_dim, num_classes=10,device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.batch_size = None
        self.hidden = None
        self.input_dim = input_dim
        self.seq_dim = seq_dim
        self.device = device
    
    def forward(self, x):
        #x = torch.reshape(x,(self.input_dim,self.seq_dim))
        h0, c0 = self.init_hidden(x)
        out,_ = self.rnn(x,h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if self.device == 'cuda':
            return [t.cuda() for t in (h0, c0)]
        else:
            return [t for t in (h0, c0)]




class GRUClassifier(nn.Module):
    def __init__(self, seq_dim, input_dim, hidden_dim, layer_dim, num_classes=10,device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.batch_size = None
        self.hidden = None
        self.input_dim = input_dim
        self.seq_dim = seq_dim
        self.device = device
    
    def forward(self, x):
        #x = torch.reshape(x,(self.input_dim,self.seq_dim))
        h0, c0 = self.init_hidden(x)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if self.device == 'cuda':
            return [t.cuda() for t in (h0, c0)]
        else:
            return [t for t in (h0, c0)]


