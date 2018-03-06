import torch

import torch.nn as nn

from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import torch.utils.data
from torchvision import datasets, transforms

import dlib

from math import *

import multiprocessing as mp

import numpy as np

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=10000, shuffle=True)

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.w = Parameter(torch.Tensor(input_size, output_size), requires_grad=False)
        self.b = Parameter(torch.Tensor(1, output_size), requires_grad=False)

    def forward(self, x):
        return x.mm(self.w) + self.b

class Net(nn.Module):
    def __init__(self, data_size=28*28):
        super(Net, self).__init__()
        self.linears = nn.ModuleList([Linear(data_size, data_size * 3), Linear(data_size * 3, 10)])

    def forward(self, x):
        for linear in self.linears:
            x = linear(x).sigmoid()
        return x
    
def train(params, key_order):
    torch.set_num_threads(1)
    model = Net()
    start = 0
    state_dict = model.state_dict()
    for k in key_order:
        v = state_dict[k]
        length = np.prod(v.shape)
        state_dict[k] = torch.FloatTensor(np.reshape(params[start:start+length], v.shape))
        start += length
    model.load_state_dict(state_dict)
    for data, target in train_loader:
        data = torch.cat(torch.unbind(data.squeeze(), 1),1)
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = -F.nll_loss(output, target).data[0]
        return loss

if __name__ == "__main__":
    with mp.Pool(processes=8) as pool:
        state_dict = Net().state_dict()
        key_order = [k for k in state_dict.keys()]
        length = 0
        for k in key_order:
            length += np.prod(state_dict[k].shape)
        opt = dlib.global_function_search([-10] * length,[10] * length, [False] * length)
        
        ps = []
        print("Start")
        for i in range(10):
            print("Optimize Step")
            evaluations = [opt.get_next_x() for _ in range(8)]
            params = [(e.x(),key_order) for e in evaluations]
            print("Eval Start")
            results = pool.starmap_async(train, params)
            print("Evaling")
            results = results.get()
            print("Eval Stop")
            for i2 in range(len(results)):
                evaluations[i2].set(-results[i2])
            print("Set done")
            x,y = opt.get_best_function_eval()
            
            print(-1*y)

        x,y = opt.get_best_function_eval()

        print(-1*y)
