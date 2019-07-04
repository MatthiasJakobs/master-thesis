import torch.nn as nn

from deephar.blocks import *

class Mpii_Small_No_Context(nn.Module):
    def __init__(self):
        super(Mpii_Small_No_Context, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=0)

    def forward(self, x):
        a = self.stem(x)
        output , _ = self.rec1(a)
        
        return output

class Mpii_No_Context(nn.Module):
    def __init__(self):
        super(Mpii_No_Context, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=0)
        self.rec2 = ReceptionBlock(num_context=0)
        self.rec3 = ReceptionBlock(num_context=0)
        self.rec4 = ReceptionBlock(num_context=0)
        self.rec5 = ReceptionBlock(num_context=0)
        self.rec6 = ReceptionBlock(num_context=0)
        self.rec7 = ReceptionBlock(num_context=0)
        self.rec8 = ReceptionBlock(num_context=0)


    def forward(self, x):
        a = self.stem(x)
        pose1 , output1 = self.rec1(a)
        pose2 , output2 = self.rec2(output1)
        pose3 , output3 = self.rec3(output2)
        pose4 , output4 = self.rec4(output3)
        pose5 , output5 = self.rec5(output4)
        pose6 , output6 = self.rec6(output5)
        pose7 , output7 = self.rec7(output6)
        pose8 , _ = self.rec8(output7)

        
        return torch.cat((pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8), 0)
