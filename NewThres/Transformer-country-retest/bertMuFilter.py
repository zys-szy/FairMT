import os
import nltk
import torch
import math
import torch.nn.functional as F
from stanfordcorenlp import StanfordCoreNLP
import random
from tqdm import tqdm
from copy import deepcopy
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

K_Number = 20

#tokenizer = TreebankWordTokenizer()
#detokenizer = TreebankWordDetokenizer()

#nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", port=34139, lang="en")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bertmodel_pre = BertModel.from_pretrained("bert-large-uncased")#'/data/szy/bertlarge')
    #bertmodel = BertModel.from_pretrained("bert-large-uncased")#'/data/szy/bertlarge')
    #bertmodel_pre = BertForMaskedLM.from_pretrained("bert-large-uncased")#'/data/szy/bertlarge')
    #berttokenizer = RobertaTokenizer.from_pretrained('bert-large-uncased')
    #bertmodel = RoBertaForMaskedLM.from_pretrained('/data/szy/bertlarge')
    bertmodel = bertmodel_pre.cuda()
    bertmodel_pre.eval()
    #bertmodel.eval()
    return bertmodel_pre, berttokenizer#, bertmodel

tokenizer = TreebankWordTokenizer()

def BertM (bert, berttoken, inpori, inpmut):
    #sentence = inpori
    ret = -1
    
    oritokens = ['[CLS]'] + inpori.split() + ['[SEP]'] #tokenizer.tokenize(sentence)
    ori_onehot = berttoken.convert_tokens_to_ids(oritokens)
    orih = bert(torch.tensor([ori_onehot]).cuda())[0][0].tolist()
    print (berttoken.convert_ids_to_tokens(ori_onehot))
    
    muttokens = ['[CLS]'] + inpmut.split() + ['[SEP]']
    mut_onehot = berttoken.convert_tokens_to_ids(muttokens)
    muth = bert(torch.tensor([mut_onehot]).cuda())[0][0].tolist() #berttoken.convert_tokens_to_ids(muttokens)
    print (berttoken.convert_ids_to_tokens(mut_onehot))

    assert len(oritokens) == len(muttokens)

    for i in range(len(oritokens)):
        if oritokens[i] != muttokens[i]:
            print (oritokens[i])
            print (muttokens[i])
            worda = orih[i]
            wordb = muth[i]
            fz = 0
            lf = 0
            rt = 0
            
            for t in range(len(worda)):
                fz += worda[t] * wordb[t]
                lf += worda[t] ** 2
                rt += wordb[t] ** 2
            
            lf = math.sqrt(lf)
            rt = math.sqrt(rt)

            ret = fz / (lf * rt)
            return ret
            #print (worda)
            #print (wordb)
            #exit()

    return ret 
    #tokens = berttoken.tokenize(sentence)
    #tag = nlp.pos_tag(" ".join(oritokens)) 
    gen = []
    for i in range(len(tokens)):
        #i = 1
        #if tag[i][1] not in ["NNS", "JJ", "NN", "NNP", "NNPS", "CD", "NNPS"]:
        #    continue
        token = tokens[i]
        print (tokens)
        ltokens = [x.lower() for x in tokens]
        ltokens[i] = '[MASK]'
        ltokens = ['[CLS]'] + ltokens + ["[SEP]"]
        print (token)
        try:
            encoding = berttoken.convert_tokens_to_ids(ltokens)
        except:
            continue
        tensor = torch.tensor([encoding])
        pre = F.softmax(bert(tensor)[0], dim=-1)
        print (pre)
        #exit()
        topk = torch.topk(pre[0][i + 1], K_Number).cpu().numpy()#.tolist()
        value = topk[0].tolist()
        topk = topk[1].tolist()
        mut = berthidden(ori_onehot)[0][0].tolist()
        print (mut)
        exit()
        print (topk)
        topkTokens = berttoken.convert_ids_to_tokens(topk)
        print (topkTokens)
        for index in range(len(topkTokens)):
            #if value[index] <= 0.05:
            #    break
            tt = topkTokens[index]
            l = deepcopy(tokens)
            l[i] = tt
            print ("------")
            print (" ".join(oritokens))
            print (" ".join(tokens))
            print (" ".join(l))
            sen = " ".join(l).replace(" ##", "")
        
            #if check_tree(tag, sen):
            gen.append(sen)
    return " ".join(tokens).replace(" ##", ""), gen

#test = "Do you like white people ?"
#test = "I like to eat food."
f = open("./f_en_mu.txt")
lines = f.readlines()
f.close()

l = []
for i in range(len(lines)):
    if i % 2 == 0:
        l.append([lines[i].strip(), lines[i + 1].strip()])
        #l.append(lines[i].strip())

bertmodel, berttoken = bertInit()

#f9 = open("9bf_en_mu.txt", "w")
f10 = open("bf_en_mu_score.txt", "w")
#f8 = open("8bf_en_mu.txt", "w")
for i in tqdm(range(len(l))):
    line = l[i]
    #tag = nlp.pos_tag(line)
    score = BertM(bertmodel, berttoken, line[0], line[1])
    print (line[0])
    print (line[1])
    print (score)
    print ("-----------")
    if False and score > 0.8:
        print ("True")
        #for sen in gen:
        if score > 0.9:
            f9.write(line[0].strip().replace(" ##", "") + "\n")
            f9.write(line[1].strip().replace(" ##", "") + "\n")
        f8.write(line[0].strip().replace(" ##", "") + "\n")
        f8.write(line[1].strip().replace(" ##", "") + "\n")
    
    f10.write(line[0].strip() + " " + str(score) + "\n")  
    f10.write(line[1].strip() + " " + str(score) + "\n")  
        
#f9.close()
#f8.close()
f10.close()
