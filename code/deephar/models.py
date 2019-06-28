import torch.nn as nn

from deephar.blocks import *

class Mpii_No_Context(nn.Module):
    def __init__(self):
        super(Mpii_No_Context, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=0)
        self.rec2 = ReceptionBlock(num_context=0)
        self.rec3 = ReceptionBlock(num_context=0)

    def forward(self, x):
        a = self.stem(x)
        _, b = self.rec1(a)
        _, b = self.rec2(b)
        output_posereg3, b = self.rec3(b)

        return output_posereg3