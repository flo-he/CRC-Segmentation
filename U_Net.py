import torch as tc
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, feat1, feat2, feat3, feat4, feat5, input_feat=3, output_feat=3, kern_size=3,
                 actfunc=nn.LeakyReLU(inplace=True), droprate=0.1, enabledrop=True, affine=True):
        #3 input channels (rgb), 3 output classes: (cancer, normal tissue, background) 
        super(NeuralNet, self).__init__()
        # dropout on/off for training/validation
        self.use_dropout = enabledrop
        # inplace activation and dropout to save memory usage
        self.dropout = nn.Dropout(droprate, inplace=True)
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
        # output layer is a 1x1 convolution to reduce feature maps to number of output classes 
        self.conv1out = nn.Conv2d(feat1, output_feat, 1)
        self.conv11 = nn.Conv2d(feat1, feat1, kern_size, padding = 1)  #no reature increase
        self.conv22 = nn.Conv2d(feat2, feat2, kern_size, padding = 1)
        self.conv33 = nn.Conv2d(feat3, feat3, kern_size, padding = 1)
        self.conv44 = nn.Conv2d(feat4, feat4, kern_size, padding = 1)
        self.conv55 = nn.Conv2d(feat5, feat5, kern_size, padding = 1)
        self.upconv1 = nn.ConvTranspose2d(feat5,feat4 , 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(feat4,feat3 , 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(feat3,feat2 , 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(feat2,feat1 , 2, stride=2)
        self.inst_norm1 = nn.InstanceNorm2d(feat1, affine=affine)
        self.inst_norm2 = nn.InstanceNorm2d(feat2, affine=affine)
        self.inst_norm3 = nn.InstanceNorm2d(feat3, affine=affine)
        self.inst_norm4 = nn.InstanceNorm2d(feat4, affine=affine)
        self.inst_norm5 = nn.InstanceNorm2d(feat5, affine=affine)
        self.inst_normout = nn.InstanceNorm2d(output_feat, affine=affine)
        self.activation = actfunc
        
        
    def forward(self, x):
        # forward pass has dropout option
        if self.use_dropout:
            return self.forw_drop(x)
        else:
            return self.forw_no_drop(x)

    def forw_drop(self, x):
        #way down
        x = self.convin1(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm1(x)) # 1. increasing features
        x = self.conv11(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm1(x)) # 2. no increase
        copy1 = x.clone()  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field

        x = self.conv12(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm2(x)) # 1. increasing features
        x = self.conv22(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm2(x)) # 2. no increase
        copy2 = x.clone()  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv23(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm3(x)) # 1. increasing features
        x = self.conv33(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm3(x)) # 2. no increase
        copy3 = x.clone()  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv34(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm4(x)) # 1. increasing features
        x = self.conv44(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm4(x)) # 2. no increase
        copy4 = x.clone()  #make copy to concatenate later
        # bottleneck
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv45(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm5(x)) # 1. increasing features
        x = self.conv55(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm5(x)) # 2. no increase

        #way up:
        x = self.upconv1(x)
        # skip connection
        x = tc.cat((copy4, x), dim=1)
        # delete copy to save GPU mem
        del copy4  
        x = self.conv54(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm4(x))
        x = self.conv44(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm4(x))
        x = self.upconv2(x)
        x = tc.cat((copy3, x), dim=1) 
        del copy3 
        x = self.conv43(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm3(x))
        x = self.conv33(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm3(x))
        x = self.upconv3(x)
        x = tc.cat((copy2, x), dim=1) 
        del copy2
        x = self.conv32(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm2(x))
        x = self.conv22(x) 
        x = self.dropout(x)
        x = self.activation(self.inst_norm2(x))
        x = self.upconv4(x)
        x = tc.cat((copy1, x), dim=1)
        del copy1
        x = self.conv21(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm1(x))
        x = self.conv11(x)
        x = self.dropout(x)
        x = self.activation(self.inst_norm1(x))
        x = self.conv1out(x)
        x = self.dropout(x)
        x = self.activation(self.inst_normout(x))
        # output is images are  3x512x512, crop to ground truth output masks 500x500
        return x[:, :, 6:506, 6:506]
    
    def forw_no_drop(self, x):
        #way down
        x = self.convin1(x)
        x = self.activation(self.inst_norm1(x)) # 1. increasing features
        x = self.conv11(x)
        x = self.activation(self.inst_norm1(x)) # 2. no increase
        copy1 = x.clone()  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field

        x = self.conv12(x)
        x = self.activation(self.inst_norm2(x)) # 1. increasing features
        x = self.conv22(x)
        x = self.activation(self.inst_norm2(x)) # 2. no increase
        copy2 = x.clone()  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv23(x)
        x = self.activation(self.inst_norm3(x)) # 1. increasing features
        x = self.conv33(x)
        x = self.activation(self.inst_norm3(x)) # 2. no increase
        copy3 = x.clone()  #make copy to concatenate later
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv34(x)
        x = self.activation(self.inst_norm4(x)) # 1. increasing features
        x = self.conv44(x)
        x = self.activation(self.inst_norm4(x)) # 2. no increase
        copy4 = x.clone()  #make copy to concatenate later
        # bottleneck
        x = self.pool(x)  #reduce resolution, increase receptive field
        x = self.conv45(x)
        x = self.activation(self.inst_norm5(x)) # 1. increasing features
        x = self.conv55(x)
        x = self.activation(self.inst_norm5(x)) # 2. no increase

        #way up:
        x = self.upconv1(x)
        # skip connection
        x = tc.cat((copy4, x), dim=1)
        # delete copy to save GPU mem
        del copy4  
        x = self.conv54(x)
        x = self.activation(self.inst_norm4(x))
        x = self.conv44(x)
        x = self.activation(self.inst_norm4(x))
        x = self.upconv2(x)
        x = tc.cat((copy3, x), dim=1) 
        del copy3 
        x = self.conv43(x)
        x = self.activation(self.inst_norm3(x))
        x = self.conv33(x)
        x = self.activation(self.inst_norm3(x))
        x = self.upconv3(x)
        x = tc.cat((copy2, x), dim=1) 
        del copy2
        x = self.conv32(x)
        x = self.activation(self.inst_norm2(x))
        x = self.conv22(x) 
        x = self.activation(self.inst_norm2(x))
        x = self.upconv4(x)
        x = tc.cat((copy1, x), dim=1)
        del copy1
        x = self.conv21(x)
        x = self.activation(self.inst_norm1(x))
        x = self.conv11(x)
        x = self.activation(self.inst_norm1(x))
        x = self.conv1out(x)
        x = self.activation(self.inst_normout(x))
        # output is images are  3x512x512, crop to ground truth output masks 500x500
        return x[:, :, 6:506, 6:506]


if __name__ == "__main__":

    test = NeuralNet(64, 128, 256, 512, 1024)
    a = test(tc.ones(size=(1, 3, 512, 512)))
    print(a.size()) # should be (1, 3, 500, 500)