import torch
from torch import nn
from torch.nn import functional as F

## Helper class to create Conv Layers
class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
    
## Helper class to create Conv Layers with batch normalization and PReLU Non-linearlty if needed    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        input_x = x
        input_x = self.conv(input_x)
        if self.bn is not None:
            input_x = self.bn(input_x)
        if self.relu is not None:
            input_x = self.relu(input_x)
        return input_x

    
## Channel Attention Module (CAM)      
class CAM(nn.Module):
    def __init__(self, channel, reduction = 8):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel// reduction , bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
       
        y = self.avg_pool(x).view(b, c)
         
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

## This module is part of  Contextual-Attention Refinement Module (CRM)
class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
     

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out
    
# Contextual-Attention Refinement Module (CRM)    
class CRM(nn.Module):
     def __init__(self,in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False):
         super(CRM,self).__init__()
         self.conv1_1x1 = nn.Conv2d(in_planes, out_planes, 1, 1)
         self.conv1_3x3 = BasicConv(in_planes = 2*in_planes, out_planes=out_planes, kernel_size=kernel_size,\
                               stride=stride, padding=padding, dilation=dilation, groups=1, relu=relu, bn=bn, bias=bias)
         self.conv12_3x3 = BasicConv(in_planes = 2*in_planes, out_planes=out_planes, kernel_size=kernel_size,\
                               stride=stride, padding=padding, dilation=dilation, groups=1, relu=relu, bn=bn, bias=bias)    
         
         self.relu1 = nn.PReLU()
         self.relu2 = nn.PReLU()
         self.se1 = CAM(channel=out_planes)
         #self.context1 =   ContextBlock(input_channel= in_planes, output_channel=out_planes, square=False)
         
         self.conv2_1x1 = nn.Conv2d(in_planes, out_planes, 1, 1)
         self.conv2_3x3 = BasicConv(in_planes = in_planes, out_planes=out_planes, kernel_size=kernel_size,\
                               stride=stride, padding=padding, dilation=dilation, groups=1, relu=relu, bn=bn, bias=bias)
         self.conv22_3x3 = BasicConv(in_planes = in_planes, out_planes=out_planes, kernel_size=kernel_size,\
                               stride=stride, padding=padding, dilation=dilation, groups=1, relu=relu, bn=bn, bias=bias)    
         self.se2 = CAM(channel=out_planes)
         self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
         self.context2 =   ContextBlock(input_channel=in_planes, output_channel=out_planes, square=False)
        
         self.fusion = BasicConv(in_planes = 2*in_planes, out_planes=out_planes, kernel_size=kernel_size,\
                                stride=stride, padding=padding, dilation=dilation, groups=1, relu=relu, bn=bn, bias=bias)
         
     def forward(self,x):
         ######## Local Residual network (DownStream Level) #######################
         x_2 = F.avg_pool2d(x, 2, 2)
         out2_1x1 = self.relu1(self.conv2_1x1(x_2))
         out2_3x3 = self.conv2_3x3(x_2)
         ## UpStream Link (1) 
         up2_3x3  = F.interpolate(out2_3x3, scale_factor=2, mode='bilinear')
         out22_3x3 = self.conv22_3x3(out2_3x3)
         ## UpStream Link (2)
         up22_3x3  = F.interpolate(out22_3x3, scale_factor=2, mode='bilinear')
         out22_3x3 = self.se2(out22_3x3)
         out2 = out22_3x3 + out2_1x1
         ## UpStream Link (3)
         out2 = self.context2(out2)
         ######## Local Residual network (MainStream Level) #######################
         out1_1x1 = self.relu2(self.conv1_1x1(x))
         ## Fusion of mainstream and Upstream link (1)
         feat_concat1 = torch.cat([x,up2_3x3],1)
         out1_3x3    = self.conv1_3x3(feat_concat1)
         ## Fusion of mainstream and Upstream link (2)
         feat_concat2 = torch.cat([out1_3x3,up22_3x3],1)
         out2_3x3    = self.conv12_3x3(feat_concat2)
         out = self.se1(out2_3x3)
         out = out + out1_1x1
         ## Fusion of mainstream and Upstream link (3)
         out_up = F.interpolate(out2, scale_factor=2, mode='bilinear')
         ultimate_out = torch.cat([out,out_up],1)
         out = self.fusion(ultimate_out)
         out = out + x
         
         return out 

#Dual Local Residual Attention Module (DLRAM)  
class DLRAM(nn.Module):
    def __init__(self,in_planes, out_planes):
        super(DLRAM,self).__init__()

        self.conv1_1x1 =  nn.Conv2d(in_planes, out_planes, 1, 1)
        self.relu1   = nn.PReLU()
        self.se1     = CAM(in_planes)
        self.se2     = CAM(in_planes)
        self.conv11_3x3 = BasicConv(in_planes = in_planes, out_planes=out_planes, kernel_size=3,\
                              stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False) 
        self.conv12_3x3 = BasicConv(in_planes = out_planes, out_planes=out_planes, kernel_size=3,\
                               stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False)   
        
        
        self.se_cat1     = CAM(out_planes)
        self.conv2_3x3_cat = BasicConv(in_planes = out_planes, out_planes=out_planes, kernel_size=3,\
                               stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False)     
        self.conv2_1x1 =  nn.Conv2d(out_planes, out_planes, 1, 1)
        self.relu2   = nn.PReLU()
        
        self.conv21_3x3 = BasicConv(in_planes = out_planes, out_planes=out_planes, kernel_size=3,\
                              stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False) 
        
        self.conv22_3x3 = BasicConv(in_planes = out_planes, out_planes=out_planes, kernel_size=3,\
                              stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False)    
        
        self.se_cat2     = CAM(out_planes)
        self.conv23_3x3_cat = BasicConv(in_planes = out_planes, out_planes=out_planes, kernel_size=3,\
                               stride=1, padding=1, dilation=1, groups=1, relu=True, bn=False, bias=False)    
        
    def forward(self,x,x_res):
        out_at   = self.se1(x)
        out1_1x1  = self.relu1(self.conv1_1x1(x))
        out      = self.se2(x)
        out      = self.conv11_3x3(out) 
        #feat_cat1 = torch.cat([out,x_res],1)
        feat_cat1 = out+x_res
        out      = self.conv12_3x3(feat_cat1)
        #feat_cat2 = torch.cat([out,out1_1x1],1)
        feat_cat2 = out+out1_1x1
        out      = self.se_cat1(feat_cat2) 
        out      = self.conv2_3x3_cat(out)
        out2_1x1  = self.relu2(self.conv2_1x1(out))
        out = self.conv21_3x3(out)
        out_res  = out
        out      = self.conv22_3x3(out)
        #feat_cat3 = torch.cat([out,out2_1x1],1)
        feat_cat3 = out+out2_1x1
        out = self.se_cat2(feat_cat3)
        out = self.conv23_3x3_cat(out)
        out = out + out_at
        return out, out_res
            


class DLRA(nn.Module):
      def __init__(self, inplanes=3, planes=31, channels=200, n_DRBs=8):
          super(DLRA, self).__init__()
          # 3DNets
          self.input1  = BasicConv(inplanes,channels,dilation=1,padding=1)
          self.input2  = BasicConv(channels,channels,dilation=1,padding=1)
          self.denoise = CRM(channels,channels)    
          self.backbone = nn.ModuleList(
              [DLRAM(in_planes=channels, out_planes=channels) for _ in
              range(n_DRBs)])
         

          self.output_prelu2D = nn.PReLU()
          self.output_conv1 = Conv3x3(channels, planes, 3, 1)
          self.output_conv2 = Conv3x3(planes, planes, 3, 1)

      def forward(self, x):
          out = self.DRN2D(x)
          return out

      def DRN2D(self, x):
          b,c,h,w = x.shape
          input_x = x
          fea_out  = None
          out = self.input1(input_x)
          out = self.input2(out)
          out = self.denoise(out)
          residual = out
    
          for i, block in enumerate(self.backbone):
              out,residual = block(out,residual)
            
          out = self.output_conv1(self.output_prelu2D(out))
          out = self.output_conv2(out)
          return out       
    
    
if __name__ == "__main__":
     import os
     os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     input_tensor = torch.rand(1, 3, 128, 128)
     model = DLRA(3,31,80,10)
    # model = nn.DataParallel(model).cuda()
     with torch.no_grad():
        output_tensor = model(input_tensor)
     print(output_tensor.size())
     print('Parameters number is ', sum(param.numel() for param in model.parameters()))
     print(torch.__version__)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    