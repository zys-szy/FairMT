import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import nltk, string
from datasets import load_dataset
from tf_idf import *
import torch
import re
import sys
import six
import nltk
import math
import numpy as np
import collections
import unicodedata
import time
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



dataset = load_dataset('news_commentary', 'en-zh')

tttt = time.time()
pns = 0.003

def bertInit():
    berttokenizeren = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    berttokenizerzh = AutoTokenizer.from_pretrained('uer/sbert-base-chinese-nli')
    berten = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    bertzh = AutoModel.from_pretrained('uer/sbert-base-chinese-nli')
    berten.eval().cuda()#.to(torch.device("cuda:0"))
    bertzh.eval().cuda()#.to(torch.device("cuda:1"))

    return berten, bertzh, berttokenizeren, berttokenizerzh #bertmodel, berttokenizer, bertori

def BertCom(sentences, model, tokenizer):
    # Tokenize sentences
    #encoded_input = #torch.tensor(tokenizer.encode(sentences)).cuda()#tokenizer([sentences], padding=True, truncation=True, return_tensors='pt')
    encoded_input = tokenizer([sentences], padding=True, truncation=True, return_tensors='pt')
    print (encoded_input)
    encoded_input["input_ids"] = encoded_input["input_ids"].cuda()
    encoded_input["token_type_ids"] = encoded_input["token_type_ids"].cuda()
    encoded_input["attention_mask"] = encoded_input["attention_mask"].cuda()
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).data.cpu().numpy()

    #print (sentence_embeddings)
    #exit()
    return sentence_embeddings[0]

berten, bertzh, berttokenizeren, berttokenizerzh = bertInit()

#print (dataset['train']['translation'])

#with open("train.txt", "w") as ftrain:
#    with open("dev.txt", "w") as fdev:
ftrain = []
fdev = []

count = 0
for data in tqdm(dataset['train']['translation']):
    en = data['en']
    zh = data['zh']
#    print (en)
#    print (zh)
#    exit()
    count += 1 
    try:
        en = BertCom(en, berten, berttokenizeren)#en = berttokenizeren.convert_tokens_to_ids(en)
        zh = BertCom(zh, bertzh, berttokenizerzh)#en = berttokenizeren.convert_tokens_to_ids(en)
    except:
        continue

    if count == 10:
        count = 0
        fdev.append([zh, en])
    else:
        ftrain.append([zh, en])

import pickle
with open('train2.pkl', "wb") as f:
    f.write(pickle.dumps(ftrain))

with open('dev2.pkl', "wb") as f:
    f.write(pickle.dumps(fdev))

