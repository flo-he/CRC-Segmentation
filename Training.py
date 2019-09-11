import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .module import Module
from .utils import _quadruple
from .. import functional as F


class _ConstantPadNd(Module):
    __constants__ = ['padding', 'value']

    def __init__(self, value):
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self):
        return 'padding={}, value={}'.format(self.padding, self.value)

class ReflectionPad2d(_ReflectionPadNd):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)
    

class NeuralNet(nn.Module):
    def __init__(self, input_feat = 3, feat1, feat2, feat3, feat4, feat5, output_feat = 3, kern_size = 3, actfunc = nn.LeakyReLu()):
        #3 input features (rgb), 3 output features: (cancer, normal tissue, background) 
        super(NeuralNet, self).__init__()
      
        self.mirror = nn.ReflectionPad2d(6)  #  do self.morror(input) only with input
        self.pad = nn.ReflectionPad2d(1)  # do self.pad(layer) before convolution with 3x3 kernel
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.convin1 = nn.Conv2d(input_feat, feat1, kern_size)
        self.conv12 = nn.Conv2d(feat1, feat2, kern_size) #from feature size 1 to size 2
        self.conv23 = nn.Conv2d(feat2, feat3, kern_size)
        self.conv34 = nn.Conv2d(feat3, feat4, kern_size)
        self.conv45 = nn.Conv2d(feat4, feat5, kern_size)
        self.conv54 = nn.Conv2d(feat5, feat4, kern_size)
        self.conv43 = nn.Conv2d(feat4, feat3, kern_size)
        self.conv32 = nn.Conv2d(feat3, feat2, kern_size)
        self.conv21 = nn.Conv2d(feat2, feat1, kern_size)
        self.conv1out = nn.Conv2d(feat1, output_feat, kern_size)
        self.conv11 = nn.Conv2d(feat1, feat1, kern_size)
        self.conv22 = nn.Conv2d(feat2, feat2, kern_size)
        self.conv33 = nn.Conv2d(feat3, feat3, kern_size)
        self.conv44 = nn.Conv2d(feat4, feat4, kern_size)
        self.conv55 = nn.Conv2d(feat5, feat5, kern_size)
        self.upconv1 = nn.ConvTranspose2d(feat5,feat4 , 2, stride=1)
        self.upconv2 = nn.ConvTranspose2d(feat4,feat3 , 2, stride=1)
        self.upconv3 = nn.ConvTranspose2d(feat3,feat2 , 2, stride=1)
        self.upconv4 = nn.ConvTranspose2d(feat2,feat1 , 2, stride=1)
        self.activation = actfunc
        
        
     def forward(self, x):
         
        x = self.mirror(x)
        #way down:
        x = self.activation(self.convin1(self.pad(x))) # 1. increasing features
        x = self.activation(self.conv11(self.pad(x)))  # 2. no increase
        copy1 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.activation(self.conv12(self.pad(x)))
        x = self.activation(self.conv22(self.pad(x)))
        copy2 = x  #make copy to concatenate later
        x = self.pool(x)
        x = self.activation(self.conv23(self.pad(x)))
        x = self.activation(self.conv33(self.pad(x)))
        copy3 = x  #make copy to concatenate later
        x = self.pool(x)
        x = self.activation(self.conv34(self.pad(x)))
        x = self.activation(self.conv44(self.pad(x)))
        copy4 = x  #make copy to concatenate later
        x = self.pool(x)
        x = self.activation(self.conv45(self.pad(x)))
        x = self.activation(self.conv55(self.pad(x)))
        #way up:
        x = self.upconv1(x)
        x = tc.cat((copy4, x))  #maybe need to give the right dim to concatenate...
        x = self.activation(self.conv54(self.pad(x)))
        x = self.activation(self.conv44(self.pad(x)))
        x = self.upconv2(x)
        x = tc.cat((copy3, x))  #maybe need to give the right dim to concatenate...
        x = self.activation(self.conv43(self.pad(x)))
        x = self.activation(self.conv33(self.pad(x)))
        x = self.upconv3(x)
        x = tc.cat((copy2, x))  #maybe need to give the right dim to concatenate...
        x = self.activation(self.conv32(self.pad(x)))
        x = self.activation(self.conv22(self.pad(x)))
        x = self.upconv4(x)
        x = tc.cat((copy1, x))  #maybe need to give the right dim to concatenate...
        x = self.activation(self.conv21(self.pad(x)))
        x = self.activation(self.conv11(self.pad(x)))
        x = self.activation(self.conv1out(self.pad(x)))
        # ..maybe one softmax layer in the end?
        return x


def train(model, n_epochs, crit = nn.MSELoss(), opt = optim.Adam(model.parameters(), lr=learningrate) ):
    model.train()
    criterion = crit
    optimizer = opt
    for epoch in range(n_epochs):
        for i, (batch, idx) in enumerate(dataloader):
            inp, tgt = batch # load input and target
            optimizer.zero_grad()  # set gradients to zero
            pred = model.forward(inp)  # predict with model
            loss = criterion(pred,tgt)
            loss.backward()  # backpropagation
            optimizer.step()  # gradient step
        if epoch % (n_epochs/10) == 0:
            print("epoch {}, loss: {}".format(epoch, loss.item()))
    return model



# set parameters for neural network

learningrate = 0.005   # learning rate for optimizer
n_epochs = 4000         # number of training epochs
batch_size = 64        # SGD minibatch size

model1 = NeuralNet(input_dim=1, hidden_dim=7, output_dim=1, n_layers=1)
model1 = train(model1, n_epochs)
