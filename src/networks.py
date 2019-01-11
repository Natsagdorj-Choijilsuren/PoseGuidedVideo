import torch
import torch.nn as nn
from torchvision import Variable

from torch.autograd import Variable


class PoseSeqGen(nn.Module):
    
    #Encoder part
    def encoder_block(self):

        return nn.Sequential(
            nn.Conv2d(),
            nn.InstanceNorm2d()
        )

    def res_block(self):

        return nn.Sequential(
            nn.,
            nn.InstanceNorm2d(),
            nn.ReLU()
        )
    
    def __init__(self):

        super(PoseSeqGen,self).__init__()

        self.encoder1 = self.encoder()
        self.encoder2 = self.encoder()

    def __forward__(self,x):

        output = x
        output = self.encoder1(output)



        return output
        


class PoseSeqDisc(nn.Module):

    def __init__(self):

        pass

    def __forward__(self):
        pass



class PoseImgDisc(nn.Module):

    def layers(self):

        return nn.Sequential(
            nn.Conv2d(),
            nn.InstanceNorm2d(),
            nn.LeakyReLU()
        )
    
    def __init__(self):

        pass

    def __forward__(self,x):

        output = x

        



