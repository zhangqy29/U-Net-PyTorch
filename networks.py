
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_same(nn.Module):
    
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0):        
        super(Conv2d_same,self).__init__()
        self.kernel_size=kernel_size
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0)
        
    def forward(self,x):
        x=F.pad(x,(self.kernel_size//2)*4,mode='replication')
        
        return self.conv(x)
   

class double_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(double_conv,self).__init__()
        self.network=nn.Sequential(
                Conv2d_same(in_channels,out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                Conv2d_same(out_channels,out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
    
    def forward(self,x):
        
        return self.network(x)


class down(nn.Module):
    def __inti__(self,in_channels,out_channels):
        super(down,self).__init__()
        self.network=nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels,out_channels)
                )
        
    def forward(self,x):
        
        return self.network(x)
    
class up(nn.Module):
     
    def __init__(self,in_channels,out_channels):
        super(up,self).__init__()
        self.upsample=nn.ConvTranspose2d(in_channels,out_channels,2,2)
        self.double_conv=double_conv(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x2=self.upsample(x2)
        x=torch.cat([x1,x2],dim=1)
        
        return self.double_conv(x)



class UNet(nn.Module):
    
    def __init__(self,n_channels,n_class):
        super(UNet,self).__init__()
        self.input=double_conv(n_channels,64)
        self.down1=down(64,128)
        self.down2=down(128,256)
        self.down3=down(256,512)
        self.down4=down(512,1024)
        
        self.up1=up(1024,512)
        self.up2=up(512,256)
        self.up3=up(256,128)
        self.up4=up(128,64)
        self.output=nn.Conv2d(64,n_class,1)
        
    def forward(self,x):
        x1=self.input(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        
        x=self.up1(x4,x5)
        x=self.up2(x3,x)
        x=self.up3(x2,x)
        x=self.up4(x1,x)
        
        return F.log_softmax(self.output(x),dim=1)
        
        
      