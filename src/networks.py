import torch
import torch.nn as nn
from torchvision import Variable

from torch.autograd import Variable
from functools import partial

class PoseSeqGen(nn.Module):
    """
    Pose Sequence Generator class
    
    """
    #Encoder part
    def encoder_block(self,ch_in,ch_out,kernel_size=4,stride=2,padding=1):

        return nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size,stride,padding),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU()
            
        )

    def res_block(self):

        return nn.Sequential(
            nn.Conv2d(),
            nn.InstanceNorm2d(),
            nn.ReLU()
        )


    def decoder_block(self):

        return nn.Sequential(
            nn.ConvTranspose2d(),
            nn.InstanceNorm2d(),
            nn.ReLU()
        )
    
    def __init__(self):

        super(PoseSeqGen,self).__init__()

        self.encoder1 = self.encoder(4,10)
        self.encoder2 = self.encoder(10,20)
        self.encoder3 = self.encoder(20,)
        self.encoder4 = self.encoder()

        self.res1 = self.res_block()
        self.res2 = self.res_block()
        self.res3 = self.res_block()

        self.dec1 = self.decoder_block()
        self.dec2 = self.decoder_block()
        
        
        
    def __forward__(self,x):

        #Encoder blocks
        output = x
        output = self.encoder1(output)
        output = self.encoder2(output)
        output = self.encoder3(output)
        output = self.encoder4(output)
        #residual blocks
        output = self.res1(output)
        output = self.res2(output)
        output = self.res3(output)
        output = self.res4(output)

        #decoder parts
        output = self.dec1(output)
        output = self.dec2(output)
        
        
        


        return output
        


class PoseSeqDisc(nn.Module):

    def block(self,ch_in,ch_out,kernel_size,stride,padding):

        return nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size,stride,padding),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU()
            )
    
    
    def __init__(self):

        self.conv1 = self.block()
        self.conv2 = self.block()
        self.conv3 = self.block()
        

    def __forward__(self,x):

        output =x
        output = self.conv1(output)
        output = self.conv2(output)
        
        



class PoseImgDisc(nn.Module):

    def conv_block(self,ch_in,ch_out,kernel_size=4,
                   stride=2,padding="same"):
        
        return nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size,stride,padding),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU()
        )
        
    
    def __init__(self):

        self.conv1 = conv_block()
        self.conv2 = conv_block()
        self.conv3 = conv_block()
        
        

    def __forward__(self,x):

        output = x
        output = self.conv1(output)

#helper classes
class GANLoss(nn.Module):

    def __init__(self):

        pass
    def __forward__(self):
        pass

    
class GRU_l(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):

        super(GRU_l,self).__init__()

        self.gru = nn.GRU(input_size,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)

        
    def __forward__(self,input_,hidden):

        _,hidden = self.gru(input_,hidden)
        
        

    def _initHidden(self):
        
        return Variable(torch.randn(1,N,self.hidden_size))


    
