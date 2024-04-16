import torch
import torch.utils.data as data
import pickle

class SumDataset(data.Dataset):
    def __init__(self, dataName="train"):
        if dataName == "train":
            self.data = pickle.load(open('train.pkl', "rb"))    
        elif dataName == "dev":
            self.data = pickle.load(open('dev.pkl', "rb"))    

    def __getitem__(self, offset):
        ans = []
        ans.append(self.data[offset][0])
        ans.append(self.data[offset][1])
        return ans

    def __len__(self):
        return len(self.data) #sum([len(self.data[i][f"data0"]) for i in range(10)])
