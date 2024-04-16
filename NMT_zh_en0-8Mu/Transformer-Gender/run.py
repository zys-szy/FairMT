import torch
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
import pickle
import random
from ScheduledOptim import ScheduledOptim

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'embedding_size':768,
    'seed':1
})

def save_model(model, GArulead, ruleRelation, dirs = "MapNN"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="MapNN"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))

use_cuda = torch.cuda.is_available()

def gVar(data, copy=1):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

def train():
    global args
    dev_set = SumDataset("dev")
    train_set = SumDataset("train")
    from copy import deepcopy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=10240,
                                              shuffle=True, drop_last=True, num_workers=1)
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=5024,
                              shuffle=False, drop_last=True, num_workers=1)
    model = MapNN(args)
    load_model(model)
    if torch.cuda.is_available():
        print('using GPU')
        model = model.cuda()
        torch.cuda.manual_seed_all(args.seed)
        #model = BalancedDataParallel(40, model)#nn.DataParallel(model)
        model = nn.DataParallel(model)
        #model = nn.DataParallel(model)
    
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=1e-4), args.embedding_size, 40000)
    
    minloss = 1e9
    for epoch in tqdm(range(20000001)):
        model = model.train()
        index = 0
            
        for dB in tqdm(data_loader):
            dBatch = dB #deepcopy(dB)
            losslist = []
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss = model(dBatch[0], dBatch[1])
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            index += 1
            if epoch % 1 == 0 and index == 1:
                model = model.eval()
                losses = []
                for dB in tqdm(devloader):
                  devBatch = dB
                  for i in range(len(devBatch)):
                    devBatch[i] = gVar(devBatch[i])
                  with torch.no_grad():
                    l = model(devBatch[0], devBatch[1])
                    l = l.mean()
                    losses.append(l.item())
                losses = sum(losses) / len(losses)
                print ("Now Loss is " + str(losses))
                
                if  losses < minloss:
                    minloss = losses
                    print("find better score " + str(minloss))
                    save_model(model.module, 1, 1)
        
if __name__ == "__main__":
    train()
    #test()



