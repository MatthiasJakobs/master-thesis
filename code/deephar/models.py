import torch.nn as nn

from deephar.blocks import *

class Mpii_No_Context(nn.Module):
    def __init__(self):
        super(Mpii_No_Context, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=0)

    def forward(self, x):
        a = self.stem(x)
        output , _ = self.rec1(a)
        
        return output