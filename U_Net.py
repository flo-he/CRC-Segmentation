import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
    
class NeuralNet(nn.Module):
    def __init__(self, feat1, feat2, feat3, feat4, feat5,input_feat = 3, output_feat = 3, kern_size = 3,
                 actfunc = nn.LeakyReLU(), droprate = 0.1, enabledrop = True):
        #3 input features (rgb), 3 output features: (cancer, normal tissue, background) 
        super(NeuralNet, self).__init__()
        self.use_dropout = enabledrop
        self.dropout = nn.Dropout(droprate)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.convin1 = nn.Conv2d(input_feat, feat1, kern_size, padding = 1)
        self.conv12 = nn.Conv2d(feat1, feat2, kern_size, padding = 1) #from feature size 1 to size 2
        self.conv23 = nn.Conv2d(feat2, feat3, kern_size, padding = 1)
        self.conv34 = nn.Conv2d(feat3, feat4, kern_size, padding = 1)
        self.conv45 = nn.Conv2d(feat4, feat5, kern_size, padding = 1)
        self.conv54 = nn.Conv2d(feat5, feat4, kern_size, padding = 1) #from feature size 5 to 4
        self.conv43 = nn.Conv2d(feat4, feat3, kern_size, padding = 1)
        self.conv32 = nn.Conv2d(feat3, feat2, kern_size, padding = 1)
        self.conv21 = nn.Conv2d(feat2, feat1, kern_size, padding = 1)
        self.conv1out = nn.Conv2d(feat1, output_feat, kern_size, padding = 1)
        self.conv11 = nn.Conv2d(feat1, feat1, kern_size, padding = 1)  #no reature increase
        self.conv22 = nn.Conv2d(feat2, feat2, kern_size, padding = 1)
        self.conv33 = nn.Conv2d(feat3, feat3, kern_size, padding = 1)
        self.conv44 = nn.Conv2d(feat4, feat4, kern_size, padding = 1)
        self.conv55 = nn.Conv2d(feat5, feat5, kern_size, padding = 1)
        self.upconv1 = nn.ConvTranspose2d(feat5,feat4 , 2, stride=1)
        self.upconv2 = nn.ConvTranspose2d(feat4,feat3 , 2, stride=1)
        self.upconv3 = nn.ConvTranspose2d(feat3,feat2 , 2, stride=1)
        self.upconv4 = nn.ConvTranspose2d(feat2,feat1 , 2, stride=1)
        self.bn1 = nn.BatchNorm2d(feat1)
        self.bn2 = nn.BatchNorm2d(feat2)
        self.bn3 = nn.BatchNorm2d(feat3)
        self.bn4 = nn.BatchNorm2d(feat4)
        self.bn5 = nn.BatchNorm2d(feat5)
        self.bnout = nn.BatchNorm2d(output_feat)
        self.activation = actfunc
        
        
    def forward(self, x):
        if self.use_dropout:
            return self.forw_drop(x)
        else:
            return self.forw_no_drop(x)

    def forw_drop(self, x):
        #way down
        x = self.convin1(x)
        
        x = self.dropout(x)
        x = self.activation(self.bn1(x)) # 1. increasing features
        x = self.conv11(x)
        x = self.dropout(x)
        x = self.activation(self.bn1(x)) # 2. no increase
        copy1 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        
        x = self.conv12(x)
        x = self.dropout(x)
        x = self.activation(self.bn2(x)) # 1. increasing features
        x = self.conv22(x)
        x = self.dropout(x)
        x = self.activation(self.bn2(x)) # 2. no increase
        copy2 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv23(x)
        x = self.dropout(x)
        x = self.activation(self.bn3(x)) # 1. increasing features
        x = self.conv33(x)
        x = self.dropout(x)
        x = self.activation(self.bn3(x)) # 2. no increase
        copy3 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv34(x)
        x = self.dropout(x)
        x = self.activation(self.bn4(x)) # 1. increasing features
        x = self.conv44(x)
        x = self.dropout(x)
        x = self.activation(self.bn4(x)) # 2. no increase
        copy4 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv45(x)
        x = self.dropout(x)
        x = self.activation(self.bn5(x)) # 1. increasing features
        x = self.conv55(x)
        x = self.dropout(x)
        x = self.activation(self.bn5(x)) # 2. no increase
        copy5 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        ########## Hier noch weiter dropout
        #way up:
        x = self.upconv1(x)
        x = tc.cat((copy4, x))  #maybe need to give the right dim to concatenate...
        x = self.conv54(x)
        x = self.dropout(x)
        x = self.activate(bn4(x))
        x = self.conv44(x)
        x = self.dropout(x)
        x = self.activate(bn4(x))
        x = self.upconv2(x)
        x = tc.cat((copy3, x))  #maybe need to give the right dim to concatenate...
        x = self.conv43(x)
        x = self.dropout(x)
        x = self.activate(bn3(x))
        x = self.conv33(x)
        x = self.dropout(x)
        x = self.activate(bn3(x))
        x = self.upconv3(x)
        x = tc.cat((copy2, x))  #maybe need to give the right dim to concatenate...
        x = self.conv32(x)
        x = self.dropout(x)
        x = self.activate(bn2(x))
        x = self.conv22(x) 
        x = self.dropout(x)
        x = self.activate(bn2(x))
        x = self.upconv4(x)
        x = tc.cat((copy1, x))  #maybe need to give the right dim to concatenate...
        x = self.conv21(x)
        x = self.dropout(x)
        x = self.activate(bn1(x))
        x = self.conv11(x)
        x = self.dropout(x)
        x = self.activate(bn1(x))
        x = self.conv1out(x)
        x = self.dropout(x)
        x = self.activate(bnout(x))
    
    def forw_no_drop(self, x):
        #way down
        x = self.convin1(x)
        
        x = self.dropout(x)
        x = self.activation(self.bn1(x)) # 1. increasing features
        x = self.conv11(x)
        x = self.activation(self.bn1(x)) # 2. no increase
        copy1 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv12(x)
        x = self.activation(self.bn2(x)) # 1. increasing features
        x = self.conv22(x)
        x = self.activation(self.bn2(x)) # 2. no increase
        copy2 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv23(x)
        x = self.activation(self.bn3(x)) # 1. increasing features
        x = self.conv33(x)
        x = self.activation(self.bn3(x)) # 2. no increase
        copy3 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv34(x)
        x = self.activation(self.bn4(x)) # 1. increasing features
        x = self.conv44(x)
        x = self.activation(self.bn4(x)) # 2. no increase
        copy4 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv45(x)
        x = self.activation(self.bn5(x)) # 1. increasing features
        x = self.conv55(x)
        x = self.activation(self.bn5(x)) # 2. no increase
        copy5 = x  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        #way up:
        x = self.upconv1(x)
        x = tc.cat((copy4, x))  #maybe need to give the right dim to concatenate...
        x = self.conv54(x)
        x = self.activate(bn4(x))
        x = self.conv44(x)
        x = self.activate(bn4(x))
        x = self.upconv2(x)
        x = tc.cat((copy3, x))  #maybe need to give the right dim to concatenate...
        x = self.conv43(x)
        x = self.activate(bn3(x))
        x = self.conv33(x)
        x = self.activate(bn3(x))
        x = self.upconv3(x)
        x = tc.cat((copy2, x))  #maybe need to give the right dim to concatenate...
        x = self.conv32(x)
        x = self.activate(bn2(x))
        x = self.conv22(x) 
        x = self.activate(bn2(x))
        x = self.upconv4(x)
        x = tc.cat((copy1, x))  #maybe need to give the right dim to concatenate...
        x = self.conv21(x)
        x = self.activate(bn1(x))
        x = self.conv11(x)
        x = self.activate(bn1(x))
        x = self.conv1out(x)
        x = self.activate(bnout(x))
        return x

# noch erledigen: 
# testen mit beispiel tensoren
# insbesondere skip connection

if __name__ == "__main__":

    test = NeuralNet()  # needs to be initialized with the right parameters
    a = test(tc.ones(size=(1, 3, 512, 512)))#
    print(a.size()) # should be (1, 3, 500, 500)