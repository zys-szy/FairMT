import nltk
from tqdm import tqdm
import numpy as np
import string
import math
import torch
from multiprocessing import Process
import sys
import torch.nn.functional as F
#from stanfordcorenlp import StanfordCoreNLP
import os
from flair.data import Sentence
from flair.models import SequenceTagger
import time
import random
from copy import deepcopy
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM, ElectraForPreTraining, ElectraTokenizerFast, ElectraModel
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from GenderMutantGeneration import GenderMutantGeneration
from CountryMutantGeneration import CountryMutantGeneration
from utils import preprocessText

#os.environ["CUDA_VISIBLE_DEVICES"]="6"

K_Number = 100
Max_Mutants = 5

ft = time.time()
tker = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

#nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", port=34141, lang="en")

def check_tree (ori_tag, line):
    tag = line.strip()
    tag = nlp.pos_tag(tag)
    #print (tag)
    #print (ori_tag)
    #print ("-----------------")
    if len(tag) != len(ori_tag):
        return False
    for i in range(len(tag)):
        if tag[i][1] != ori_tag[i][1]:
            return False
    return True


def bertInit():
    #config = Ber
    berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")#'/data/szy/bertlarge')
    bertori = BertModel.from_pretrained("bert-large-cased")#'/data/szy/bertlarge')
    #berttokenizer = RobertaTokenizer.from_pretrained('bert-large-uncased')
    #bertmodel = RoBertaForMaskedLM.from_pretrained('/data/szy/bertlarge')
    electratokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    electra = ElectraModel.from_pretrained("google/electra-large-discriminator")
    
    electra.eval().cuda()
    bertmodel.eval().cuda()#.to(torch.device("cuda:0"))
    bertori.eval().cuda()#.to(torch.device("cuda:1"))
    
    return bertmodel, berttokenizer, bertori, electra, electratokenizer

tokenizer = TreebankWordTokenizer()

lcache = []

tagger = SequenceTagger.load('ner')
tagger.cuda()

def getNEPar(inpori):
    return [[] for k in inpori]
    sents = [Sentence(k) for k in inpori]
    tagger.predict(sents)
#    sentence = inpori
    ls = []
    for sent in sents:
        l = []
        for entity in sent.get_spans('ner'):
            if "PER (" in str(entity):
                continue
            l.append(entity.text)
        ls.append(l)
    return [[] for l in ls]

def getNE(inpori):
    return []
    sent = Sentence(inpori)
    tagger.predict(sent)
#    sentence = inpori
    l = []
    for entity in sent.get_spans('ner'):
        if "PER (" in str(entity):
            continue
        l.append(entity.text)
    return []

def read2Mask():
    l = []
    kk = []
    dic = {}
    with open("./jobs.txt") as f:
        lines = f.readlines()
        l = []
        l += [line.strip() for line in lines]
        l = [k.split() for k in l]
        kk += l
        for q in l:
            for qq in q:
                if qq not in dic:
                    dic[qq] = ["jobs"]
                else:
                    dic[qq].append("jobs")
    with open("./age.txt") as f:
        lines = f.readlines()
        l = []
        l += [line.strip() for line in lines]
        l = [k.split() for k in l]
        for k in l:
            if "<PLH>" in k :
                for i in range(0,5001):
                    q = deepcopy(k)
                    q[0] = str(i)
                    l.append(q)
        kk += l
        for q in l:
            for qq in q:
                #print (q)
                #print (qq)
                if qq not in dic:
                    dic[qq] = ["age"]
                else:
                    dic[qq].append("age")
        #dic["age"] = l
    with open("./race.txt") as f:
        lines = f.readlines()
        l = []
        l += [line.strip() for line in lines]
        l = [k.split() for k in l]
        kk += l
        for q in l:
            for qq in q:
                if qq not in dic:
                    dic[qq] = ["race"]
                else:
                    dic[qq].append("race")
        #dic["race"] = l
    with open("./name.txt") as f:
        lines = f.readlines()
        l = []
        l += [line.strip().lower() for line in lines]
        l = [k.split() for k in l]
        kk += l
        for q in l:
            for qq in q:
                if qq not in dic:
                    dic[qq] = ["gender"]
                else:
                    dic[qq].append("gender")
    with open("./gender.txt") as f:
        lines = f.readlines()
        l = []
        l += [line.strip() for line in lines]
        l = [k.split() for k in l]
        kk += l
        for q in l:
            for qq in q:
                if qq not in dic:
                    dic[qq] = ["gender"]
                else:
                    dic[qq].append("gender")
        #dic["gender"] = l
    #exit()
#    for i in range(len(l)):
#        l[i] = " ".join(l[i])
    return kk, dic
encodingl = []
def BertM (bert, berttoken, inpori, bertori):
    global lcache
    global encodingl
    for k in lcache:
        if inpori == k[0]:
            return k[1], k[2]
    sentence = inpori
    
    ne = getNE(inpori)
#    print (ne)
#    if ne is None:
#        return "", []

    inputtokens = inpori.split()
    oritokens = tokenizer.tokenize(sentence)
    tokens = berttoken.tokenize(sentence)
    if len(tokens) > 400:
        raise AssertionError
    count = 0
    start = 0
    index = [0] * len(tokens)
    startlist = [0] * len(tokens)
    for i in range(len(tokens)):
        token = tokens[i]
        if "##" in token:
            index[i] = count - 1
            startlist[i] = start
        else:
            startlist[i] = i
            start = i
            index[i] = count 
            count += 1
    
    k = deepcopy(ne)
    for i in ne:
        k += i.split()
    orine = ne
    ne = k
    batchsize = 1000 // len(tokens)
    gen = []
    ltokens = ["[CLS]"] + tokens + ["[SEP]"]
#    try:
#        encoding = [[berttoken.convert_tokens_to_ids(ltokens[0:i] + ["[MASK]"] + ltokens[i + 1:]), inputtokens[index[i - 1]] in ne, [i, i + 1]] for i in range(1, len(ltokens) - 1)]#.cuda()
#    except:
#        return " ".join(tokens), gen
    used = [False] * len(tokens)
    encoding = []
    needMask, maskClass = read2Mask()
    def MaskSite(k):
        l = []
#            print (k)
        for i in range(len(k)):
            if k[i] == "[MASK]":
                l.append(i)
        return l
  
    
    def getEncoding(inpori, encodingq, index=None):
        tks = berttoken.tokenize(inpori)
        inpo = " ".join(berttoken.tokenize(inpori)).replace(" ##", "")
        tks = inpo.split()
#        print (inpori)
        mute = []
        for i in range(len(tks)):
             tk = tks[i]
             flag = True
             for t in needMask:
                 if tk.lower() in t:
                     flag = False
                     break
#             if flag:
             mute.append(flag)
        
        tks = berttoken.tokenize(inpori)
        index = -1
        for i in range(len(tks)):
            tk = tks[i]
#            print (tk)
#            print ()
            if "##" not in tk:
                index += 1
                assert index < len(tks)
            if mute[index]:
                newtokens = deepcopy(tks)
                newtokens[i] = "[MASK]"
                newltokens = ["[CLS]"] + newtokens + ["[SEP]"]
                k = berttoken.convert_tokens_to_ids(newltokens)
                l = [k, True, MaskSite(newltokens)]
               # print ()
                if l not in encodingq:
                    encodingq.append(l)

    getEncoding(inpori, encoding)
    length = len(encoding)
    print (length)
    p = []
    ma = 0
    for t in range(len(encoding)):
        ma = max(ma, len(encoding[t][0]))
    for t in range(len(encoding)):
        encoding[t][0] += [0] * (ma - len(encoding[t][0]))
    for i in range(0, len(encoding), batchsize):
        tensor = [k[0] for k in encoding[i: min(len(encoding), i + batchsize)]]
        tensor = torch.tensor(tensor).cuda()
        pre = F.softmax(bert(tensor)[0], dim=-1).data.cpu()
        p.append(pre)
    if len(p) == 0:
        return " ".join(tokens), []
    pre = torch.cat(p, 0)
    tarl = [[tokens, -1]]
    for i in range(len(encoding)):
        wordindex = encoding[i][2]
        topks = []
        values = []
        flag = True
        for index in wordindex:
            isne = encoding[i][1]
            if tokens[index - 1] in string.punctuation:
                flag = False
                continue
            topk = torch.topk(pre[i][index], K_Number)#.tolist()
            value = topk[0].numpy()
            topk = topk[1].numpy().tolist()
            topkTokens = berttoken.convert_ids_to_tokens(topk)
            topks.append(topkTokens)
            values.append(value)
        if not flag:
            continue
        assert len(wordindex) != 0
        sentences = []
        ttlist = []
        tarlcandi = []
        llist = []
        wordindexlist = []
        valuelist = []
        isnelist = []
        if len(wordindex) == 1:
            for index in range(len(topkTokens)):
                if value[index] < 0.01:
                    break
                tt = topkTokens[index]
                if tt in string.punctuation:
                    continue
  #              print (tt.strip(), tokens[wordindex[0] - 1])
                if tt.strip().lower() == tokens[wordindex[0] - 1].lower():
                    continue
                l = deepcopy(tokens)
                l = l[:wordindex[0] - 1] + [tt] + l[wordindex[0]:]
                finaltt = ""
                nowindex= wordindex[0]
                while nowindex >= 0 and "##" in tt:
                    finaltt = tt.replace("##", "") + finaltt
                    tt = l[nowindex]
                    nowindex -= 1
                finaltt = tt + finaltt
                #if finaltt in 
                flag = True
                for t in needMask:
                    if finaltt.lower() in t:
                        flag = False
                        break
                tt = finaltt
                ttlist.append([tt])
                sentences.append(" ".join(l).replace(" ##", ""))
                tarl.append([l, wordindex, value[index], isne])
           
    
    lDB = []
    ma = 0
    tensor = [berttoken.convert_tokens_to_ids(["[CLS]"] + l[0] + ["[SEP]"]) for l in tarl]
    for t in range(len(tensor)):
        ma = max(ma, len(tensor[t]))
    for i in range(0, len(tarl), batchsize):
        tensor = [berttoken.convert_tokens_to_ids(["[CLS]"] + l[0] + ["[SEP]"]) for l in tarl[i: min(i + batchsize, len(tarl))]]
        for t in range(len(tensor)):
            tensor[t] += [0] * (ma - len(tensor[t]))
        tensor = torch.tensor(tensor).cuda()
        lDB.append(bertori(tensor)[0].data.cpu().numpy())
    lDB = np.concatenate(lDB, axis=0)
            
    lDA = lDB[0]
    assert len(lDB) == len(tarl)
    tarl = tarl[1:]
    lDB = lDB[1:]
    for t in range(len(lDB)):
        cossim = 2
        flag = True
        print ("--")
        print (tarl[t][1])
        for k in range(len(tarl[t][1])):
            DB = lDB[t][tarl[t][1][k]]
            DA = lDA[tarl[t][1][k]]
            cossim = np.sum(DA * DB) / (np.sqrt(np.sum(DA * DA)) * np.sqrt(np.sum(DB * DB)))
            print (cossim)
            if cossim <= 0.7:
                flag = False
                break
        if flag:
            sen = " ".join(tarl[t][0])# + "\t!@#$%^& " + str(math.exp(value[index]))#.replace(" ##", "")
            gen.append([cossim, sen])
    if len(lcache) > 4:
        lcache = lcache[1:]    

    lcache.append([inpori, " ".join(tokens), gen])
    return " ".join(tokens), gen#.replace(" ##", ""), gen

f = open(sys.argv[1])
lines = f.readlines()
f.close()

l = []
for i in range(len(lines)):
    l.append(lines[i].strip())

bertmodel, berttoken, bertori, electra, electratokenizer = bertInit()

def detk(sent):
    return " ".join(tker.tokenize(detokenizer.detokenize(sent.split())))

def tk(sent):
    return " ".join(tker.tokenize(sent))

countct = 0
countgd = 0
countb = 0
f = open(sys.argv[2], "w")
fline = open(sys.argv[3], "w")
for i in tqdm(range(len(l))):
    line = l[i]
    text = preprocessText(line)
#    print (mg2.getMutants())
    count = 0
    print ("=[===-=-=-=-=-=-=-")
    #continue
    #tag = nlp.pos_tag(line)
#    try:
    tar, gen = BertM(bertmodel, berttoken, line, bertori)
#    except:
#        continue
    gen = sorted(gen)[::-1]
    count = 0
    for sen in gen:
        sen[1] = sen[1].replace(" ##", "")
        print ("Output: ")
        print (tar)
        print (sen)
        li = [detk(l[i].strip()), detk(sen[1].strip())]
        f.write(detk(tar.replace(" ##", "")).strip() + "\n")
        f.write(tk(sen[1]).strip() + "\n")
#        f.flush()
        fline.write(str(i) + "\n")
        count += 1
        countb += 1
        if count >= Max_Mutants:
            break
    count = 0
    usedd = []
    dic = {}
f.close()
fline.close()
print (countgd, countct, countb)
print (time.time() - ft)
