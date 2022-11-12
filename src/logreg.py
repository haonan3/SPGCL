import torch
import torch.nn as nn
import torch.nn.functional as F


class LogReg(nn.Module):
    def __init__(self, ft_in, hidden, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        self.fc.bias.data.fill_(0.0)


    def forward(self, seq):
        ret = self.fc(seq)
        return ret