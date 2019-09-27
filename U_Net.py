import torch as tc
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_kernel, padding, droprate=.0, Norm=nn.InstanceNorm2d, Activation=nn.LeakyReLU):
        super(ConvBlock, self).__init__()
        # in/out channel
        self.in_ch = in_channel
        self.out_ch = out_channel
        # layers within conv block
        self.Dropout = nn.Dropout(droprate, inplace=True)
        self.Norm = Norm(out_channel, affine=True)
        self.Activ = Activation(inplace=True)
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=conv_kernel, padding=padding)
        self.Conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=conv_kernel, padding=padding)

        # whole conv block 
        self.Block = nn.Sequential(
            self.Conv1,
            self.Dropout,
            self.Norm,
            self.Activ,
            self.Conv2,
            self.Dropout,
            self.Norm,
            self.Activ
        )

    def forward(self, x):
        x = self.Block(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, droprate=.0, Norm=nn.InstanceNorm2d, Activation=nn.LeakyReLU):
        super(OutputBlock, self).__init__()
        # in and output channel
        self.in_ch = in_channel
        self.out_ch = out_channel

        # forward block
        self.Block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            nn.Dropout2d(droprate, inplace=True),
            Norm(out_channel, affine=True),
            Activation(inplace=True)
        )

    def forward(self, x):
        return self.Block(x)

class UNet(nn.Module):
    def __init__(self, ch_level1, ch_level2, ch_level3, ch_level4, input_ch=3, output_ch=3, droprate=.0, Norm=nn.InstanceNorm2d, Activation=nn.LeakyReLU):
        #3 input channel (rgb), 3 output classes: (cancer, normal tissue, background) 
        super(UNet, self).__init__()
        self.skips = []
        # U_net
        self.ConvBlock_in_1 = ConvBlock(input_ch, ch_level1, 3, 1, droprate, Norm, Activation)
        self.ConvBlock_1_2 = ConvBlock(ch_level1, ch_level2, 3, 1, droprate, Norm, Activation)
        self.ConvBlock_2_3 = ConvBlock(ch_level2, ch_level3, 3, 1, droprate, Norm, Activation)
        self.Bottleneck = ConvBlock(ch_level3, ch_level4, 3, 1, droprate, Norm, Activation)
        
        self.ConvBlock_4_3 = ConvBlock(ch_level4, ch_level3, 3, 1, droprate, Norm, Activation)
        self.ConvBlock_3_2 = ConvBlock(ch_level3, ch_level2, 3, 1, droprate, Norm, Activation)
        self.ConvBlock_2_1 = ConvBlock(ch_level2, ch_level1, 3, 1, droprate, Norm, Activation)

        self.Pooling = nn.MaxPool2d(kernel_size=2)

        self.ConvT1 = nn.ConvTranspose2d(ch_level4, ch_level3, kernel_size=2, stride=2)
        self.ConvT2 = nn.ConvTranspose2d(ch_level3, ch_level2, kernel_size=2, stride=2)
        self.ConvT3 = nn.ConvTranspose2d(ch_level2, ch_level1, kernel_size=2, stride=2)

        self.OutBlock = OutputBlock(ch_level1, output_ch, 1, droprate, Norm, Activation)
        
        
    def forward(self, x):
        # level 1
        x = self.ConvBlock_in_1(x)
        self.skips.append(x.clone())
        x = self.Pooling(x)
        # level 2
        x = self.ConvBlock_1_2(x)
        self.skips.append(x.clone())
        x = self.Pooling(x)
        # level 3
        x = self.ConvBlock_2_3(x)
        self.skips.append(x.clone())
        x = self.Pooling(x)

        # bottleneck
        x = self.Bottleneck(x)

        # level 3
        x = self.ConvT1(x)
        x = tc.cat([self.skips.pop(), x], dim=1)
        x = self.ConvBlock_4_3(x)
        # level 2
        x = self.ConvT2(x)
        x = tc.cat([self.skips.pop(), x], dim=1)
        x = self.ConvBlock_3_2(x)
        # level 1
        x = self.ConvT3(x)
        x = tc.cat([self.skips.pop(), x], dim=1)
        x = self.ConvBlock_2_1(x)

        return self.OutBlock(x)

        


if __name__ == "__main__":

    unet = UNet(64, 128, 256, 512, 3, 3, .5)
    unet.train()

    a = tc.zeros(size=(1, 3, 256, 256))
    print(unet(a).size())