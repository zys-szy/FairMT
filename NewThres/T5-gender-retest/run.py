from tqdm import tqdm
from copy import deepcopy
import torch
from transformers import AutoTokenizer, AutoModel, MT5ForConditionalGeneration
import torch.utils.data as data
from torch import optim, nn
import os
import sys
import pickle 
import numpy as np

from torch.nn.parallel import DataParallel
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        #print('len(inputs): ', str(len(inputs)))
        #print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        #print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        #print (inputs)
        if len(inputs) > 0:
            bsz = inputs[0].size(self.dim)
        elif kwargs:
            bsz = list(kwargs.values())[0].size(self.dim)
        else:
            raise ValueError("You must pass inputs to the model!")
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        #print('bsz: ', bsz)
        #print('num_dev: ', num_dev)
        #print('gpu0_bsz: ', gpu0_bsz)
        #print('bsz_unit: ', bsz_unit)
        #print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


#import argparse

#p#arser = argparse.ArgumentParser()

#parser.add_argument("--local_rank", type=int)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
#argss = parser.parse_args()
args = dotdict({
    'batch_size':24,
})
#torch.distributed.init_process_group(backend="nccl")
#torch.cuda.set_device(argss.local_rank)
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        #self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class Dataset(data.Dataset):
    def __init__(self, tknz, dataName):
        self.tknz = tknz
        self.fpath = f"{dataName}"
        self.data = []
        if os.path.exists(f"{dataName}.pkl"):
            self.data = pickle.load(open(f"{dataName}.pkl", "rb"))
        else:
            self.preProcessData()

    def preProcessData(self):
        with open(f"{self.fpath}.txt") as f:
            lines = f.readlines()
        count = 0
        for i in tqdm(range(0, len(lines), 2)):
            inp = self.tknz("translate English to Chinese: " + lines[i], padding="max_length", max_length=300, truncation=True, return_tensors="pt")
            print (lines[i] )
            print ("str:" + lines[i + 1])
            outp = self.tknz(lines[i + 1], padding="max_length", max_length=300, truncation=True, return_tensors="pt").input_ids
            print (self.tknz.decode(outp[0]))
#            outp = outp.input_ids
#            labels[labels == tokenizer.pad_token_id] = -100
            outp[outp == self.tknz.pad_token_id] = -100
            self.data.append({"input_ids":inp.input_ids[0], "attention_mask": inp.attention_mask[0], "labels":outp[0]})
        
        with open(f"{self.fpath}.pkl", "wb") as f:
            f.write(pickle.dumps(self.data))

    def __getitem__(self, offset):
        return self.data[offset]

    def __len__(self):
 #       for key in self.data:
        return len(self.data)

def initModel(tknzpath=None, modelpath=None):
    if tknzpath is not None:
        tknz = AutoTokenizer.from_pretrained(tknzpath)
    else:
        tknz = AutoTokenizer.from_pretrained("google/mt5-small")
    
    if modelpath is not None:
        model = MT5ForConditionalGeneration.from_pretrained(modelpath)
    else:
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    return tknz, model

def gVar(data):
#    print (data)
#    exit()
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    #if use_cuda:
    tensor = tensor.cuda()
    return tensor

def train():
    tknz, model = initModel(tknzpath="./model/model", modelpath="./model/model")
    trainingSet = Dataset(tknz, "train")
    #train_sampler = torch.utils.data.distributed.DistributedSampler(trainingSet)
    devSet = Dataset(trainingSet.tknz, "dev")
    devSet.tknz.save_pretrained("./model/model")
    #dev_sampler = torch.utils.data.distributed.DistributedSampler(devSet)
#    exit()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = ScheduledOptim(optimizer, d_model=768, n_warmup_steps=4000)
    maxAcc = 0
    model = model.cuda()
    model = BalancedDataParallel(0, model)
    model = model.train()
   
    for epoch in range(1000000):
        j = 0
        data_loader = torch.utils.data.DataLoader(dataset=trainingSet, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=1)
        if j == 0:
            dev_data_loader = torch.utils.data.DataLoader(dataset=devSet, batch_size=args.batch_size // 2, shuffle=False, drop_last=True, num_workers=1)
            accs = []
            acc2 = 0
            with torch.no_grad():
                model = model.eval()
                for batch in tqdm(dev_data_loader): 
                    for it in batch:
                        batch[it] = gVar(batch[it])
                    output = model(**batch)
                    print (output.logits.size())
                    logits = output.logits.argmax(-1).data.cpu().numpy()
                    labels = batch["labels"].data.cpu().numpy()
                    for t in range(len(logits)):
                        ac = 0
                        count = 0
                        for i in range(len(logits[t])):
                            if logits[t][i] == labels[t][i]:
                                ac += 1
                            if labels[t][i] != -100:
                                count += 1
                        accs.append(ac / count)
                        if ac == count:
                            acc2 += 1
            model = model.train()
            acc = sum(accs) / len(accs)
            if acc2 >= maxAcc:
                maxAcc = acc2
                print (f"Best Acc(step):{acc} Acc:{acc2}")
                model.module.save_pretrained("./model/model")
        pbar = tqdm(data_loader)
        for batch in pbar: 
            for it in batch:
                batch[it] = gVar(batch[it])
            output = model(**batch)
            loss = output.loss.mean()
            pbar.set_description(f"Loss {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
        j += 1

def gen():
    from transformers import pipeline, set_seed
    generator = pipeline('text-generation', model='./model/model')
    output = generator("1921 or 1932?", max_length=511, num_return_sequences=1)
    print (output)

if __name__ == "__main__":
    #if sys.argv[1] == "train":
    train()
    #elif sys.argv[1] == "gen":
    #    gen()
