################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

def blockUNet(in_c, out_c, name, conv_type, bn=True, relu=True, size=3, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if conv_type == 1:
        #block.add_module('%s_conv' % name, nn.Conv3d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
        block.add_module('%s_conv' % name, nn.Conv3d(in_c, out_c, kernel_size=size, stride=1, padding=pad, bias=True))
        block.add_module('%s_maxpool' % name, nn.MaxPool3d(kernel_size=2, stride=2))
    elif conv_type == 2:            
        block.add_module('%s_conv' % name, nn.Conv3d(in_c, out_c, kernel_size=size, stride=1, padding=pad, bias=True))
    elif conv_type == 3:
        #block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        # bilinear mode needs 4D input
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='trilinear'))
        block.add_module('%s_tconv' % name, nn.Conv3d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm3d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout3d( dropout, inplace=True))
    return block
    
# generator model
class TurbNetG(nn.Module):
    def compute_L1_loss(self, w):
        return torch.abs(w).sum()
    
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        
        self.layer1.add_module('layer1_conv', nn.Conv3d(9, channels, 3, 1, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', 1, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', 1, bn=True,  relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', 1, bn=True,  relu=False, dropout=dropout )
        self.layer5 = blockUNet(channels*8, channels*16, 'layer4', 1, bn=True,  relu=False, dropout=dropout )
        self.layer6 = blockUNet(channels*16, channels*32, 'layer4', 1, bn=True,  relu=False, dropout=dropout )
        
        self.dlayer6 = blockUNet(channels*32,channels*16, 'dlayer4', 3, bn=True, relu=True, dropout=dropout, size=4)         
        self.dlayer6_layer5 = blockUNet(channels*32,channels*16, 'dlayer4_layer3', 2, bn=True, relu=True, dropout=dropout, size=3) 
        
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer4', 3, bn=True, relu=True, dropout=dropout, size=4)         
        self.dlayer5_layer4 = blockUNet(channels*16,channels*8, 'dlayer4_layer3', 2, bn=True, relu=True, dropout=dropout, size=3) 
        
        self.dlayer4 = blockUNet(channels*8,channels*4, 'dlayer4', 3, bn=True, relu=True, dropout=dropout, size=4)         
        self.dlayer4_layer3 = blockUNet(channels*8,channels*4, 'dlayer4_layer3', 2, bn=True, relu=True, dropout=dropout, size=3) 
        
        self.dlayer3 = blockUNet(channels*4, channels*2, 'dlayer3', 3, bn=True, relu=True, dropout=dropout, size=4)
        self.dlayer3_layer2 = blockUNet(channels*4, channels*2, 'dlayer3_layer2', 2, bn=True, relu=True, dropout=dropout, size=3)
        
        self.dlayer2 = blockUNet(channels*2, channels, 'dlayer2', 3, bn=True, relu=True, dropout=dropout, size=4)
        self.dlayer2_layer1 = blockUNet(channels*2, channels, 'dlayer2_layer1', 2, bn=True, relu=True, dropout=dropout, size=3)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        # self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose3d(channels*2, 3, 4, 2, 1, bias=True))
                
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose3d(channels, 1, 3, 1, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        #print('out1.shape = ', out1.shape)
        out2 = self.layer2(out1)
        #print('out2.shape = ', out2.shape)
        out3 = self.layer3(out2)
        #print('out3.shape = ', out3.shape)
        out4 = self.layer4(out3)
        #print('out4.shape = ', out4.shape)
        out5 = self.layer5(out4)
        #print('out5.shape = ', out5.shape)
        out6 = self.layer6(out5)
        #print('out6.shape = ', out6.shape)
        
        dout6 = self.dlayer6(out6)
        #print('dout6.shape = ', dout6.shape)
        dout6_out5 = torch.cat([dout6, out5], 1)
        #print('dout6_out5.shape = ', dout6_out5.shape)   
        
        dout5 = self.dlayer6_layer5(dout6_out5)
        #print('dout5.shape = ', dout5.shape)                   
        dout5 = self.dlayer5(dout5)
        #print('dout5.shape = ', dout5.shape)
        dout5_out4 = torch.cat([dout5, out4], 1)
        #print('dout5_out4.shape = ', dout5_out4.shape)   
               
        dout4 = self.dlayer5_layer4(dout5_out4)
        #print('dout4.shape = ', dout4.shape)                   
        dout4 = self.dlayer4(dout4)
        #print('dout4.shape = ', dout4.shape)
        dout4_out3 = torch.cat([dout4, out3], 1)
        #print('dout4_out3.shape = ', dout4_out3.shape)   
           
        dout3 = self.dlayer4_layer3(dout4_out3)
        #print('dout3.shape = ', dout3.shape)            
        dout3 = self.dlayer3(dout3)
        #print('dout3.shape = ', dout3.shape)
        dout3_out2 = torch.cat([dout3, out2], 1)
        #print('dout3_out2.shape = ', dout3_out2.shape)    
          
        dout2 = self.dlayer3_layer2(dout3_out2)
        #print('dout2.shape = ', dout3.shape)            
        dout2 = self.dlayer2(dout2)
        #print('dout2.shape = ', dout2.shape)
        dout2_out1 = torch.cat([dout2, out1], 1)
        #print('dout2_out1.shape = ', dout2_out1.shape)
        
        dout1 = self.dlayer2_layer1(dout2_out1)
        #print('dout1.shape = ', dout1.shape)
        dout1 = self.dlayer1(dout1)
        #print('dout1.shape = ', dout1.shape)
        return dout1

# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Module):
    def __init__(self, in_channels1, in_channels2,ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv3d(in_channels1 + in_channels2, ch, 3, stride=2, padding=2)
        self.c1 = nn.Conv3d(ch  , ch*2, 3, stride=2, padding=2)
        self.c2 = nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=2)
        self.c3 = nn.Conv3d(ch*4, ch*8, 3, stride=2, padding=2)
        self.c4 = nn.Conv3d(ch*8, 1   , 3, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm3d(ch*2)
        self.bnc2 = nn.BatchNorm3d(ch*4)
        self.bnc3 = nn.BatchNorm3d(ch*8)        

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2),1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h) 
        return h

