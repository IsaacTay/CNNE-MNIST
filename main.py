import torch

import torch.nn as nn

from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import torch.utils.data
from torchvision import datasets, transforms

import dlib

from math import *

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1000, shuffle=True)

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
        print(self.state_dict())
        for linear in self.linears:
            x = linear(x).sigmoid()
        return x
    
def train(params):
    model = Net()
    for data, target in train_loader:
        data = torch.cat(torch.unbind(data.squeeze(), 1),1)
        data, target = Variable(data), Variable(target)
        output = model(data)
        print(output[0])
        loss = F.nll_loss(output, target).data[0]
        return loss

def holder_table(x0,x1):
    return -abs(sin(x0)*cos(x1)*exp(abs(1-sqrt(x0*x0+x1*x1)/pi)))

opt = dlib.global_function_search([-10, -10],[10, 10], [False, False])

for i in range(80):
    next_x = opt.get_next_x()
    x = next_x.x()
    next_x.set(-1*holder_table(x[0], x[1]))

x,y = opt.get_best_function_eval()

print(x)
print(-1*y)

x,y = dlib.find_min_global(holder_table, [-10,-10], [10,10], 80)

print(x)
print(y)
