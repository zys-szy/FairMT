import torch.nn as nn
import torch.nn.functional as F
import torch
from LayerNorm import LayerNorm

class MapNN(nn.Module):
    def __init__(self, args):
        super(MapNN, self).__init__()
        self.embedding_size = args.embedding_size
        self.Linear1 = nn.Linear(self.embedding_size, 1024)
#        self.Linear2 = nn.Linear(1024, 1024)
        self.Linears = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(7)])
        self.Norms = nn.ModuleList([LayerNorm(1024) for _ in range(7)])
        self.Linear3 = nn.Linear(1024, self.embedding_size)
        self.dropout = nn.Dropout(0.2)
    
    def CosSim (self, v1, v2):
        return torch.sum(v1 * v2, -1) / (torch.sqrt(torch.sum(v1 ** 2, -1)) * torch.sqrt(torch.sum(v2 ** 2, -1)))

    def forward(self, inv, outv = None):
        x = inv
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        for Linear, Norm in zip(self.Linears, self.Norms):
            x = Norm(self.dropout(F.relu(Linear(x)) + x))
#            x = F.relu(x)
#            x = self.dropout(x) + x
        x = self.Linear3(x)
        
        if outv is None:
            return x

        #loss = (outv - x) ** 2 + torch.mean(1 - self.CosSim(outv, x))#-torch.log(torch.gather(res.clamp(min=1e-7, max=1.0), -1, target.unsqueeze(-1))).squeeze(-1)
        loss = torch.mean(1 - self.CosSim(outv, x))#-torch.log(torch.gather(res.clamp(min=1e-7, max=1.0), -1, target.unsqueeze(-1))).squeeze(-1)
        
        loss = loss.mean()
        return loss #totalloss, res, acc, acc2, prob
 
