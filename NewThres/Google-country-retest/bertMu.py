import nltk
import math
import torch
import torch.nn.functional as F
from stanfordcorenlp import StanfordCoreNLP
import random
from copy import deepcopy
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

K_Number = 20

#tokenizer = TreebankWordTokenizer()
#detokenizer = TreebankWordDetokenizer()

nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", port=34139, lang="en")

def check_tree (ori_tag, line):
    tag = line.strip()
    tag = nlp.pos_tag(tag)
#    print (str(tag))
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
    #berttokenizer = RobertaTokenizer.from_pretrained('bert-large-uncased')
    #bertmodel = RoBertaForMaskedLM.from_pretrained('/data/szy/bertlarge')
    bertmodel.eval().cuda()
    return bertmodel, berttokenizer

tokenizer = TreebankWordTokenizer()

def BertM (bert, berttoken, inpori):
    sentence = inpori
#    oritokens = tokenizer.tokenize(sentence)
#    tokens = berttoken.tokenize(sentence)
    tag = nlp.pos_tag(" ".join(sentence))
    print (str(tag))
    exit()
    for i in range(len(
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
            encoding = berttoken.convert_tokens_to_ids(ltokens)#.cuda()
        except:
            continue
        tensor = torch.tensor([encoding]).cuda()
        pre = F.softmax(bert(tensor)[0], dim=-1)
        print (pre)

        topk = torch.topk(pre[0][i + 1], K_Number)#.tolist()
        value = topk[0].data.cpu().numpy().tolist()
        topk = topk[1].data.cpu().numpy().tolist()
        
        print (topk)
        topkTokens = berttoken.convert_ids_to_tokens(topk)
        print (topkTokens)
        for index in range(len(topkTokens)):
            if value[index] <= 0.01:
                break
            tt = topkTokens[index]
            l = deepcopy(tokens)
            l[i] = tt
            print ("------")
            print (" ".join(oritokens))
            print (" ".join(tokens))
            print (" ".join(l))
            sen = " ".join(l) #+ "\t!@#$%^& " + str(value[index])#.replace(" ##", "")
        
            #if check_tree(tag, sen):
            gen.append(sen)
    return " ".join(tokens), gen#.replace(" ##", ""), gen

#test = "Do you like white people ?"
#test = "I like to eat food."
f = open("./en_tk.txt")
lines = f.readlines()
f.close()

l = []
for i in range(len(lines)):
    #if i % 2 == 1:
    l.append(lines[i].strip())

bertmodel, berttoken = bertInit()

f = open("en_mu.txt", "w")
for i in range(len(l)):
    line = l[i]
    #tag = nlp.pos_tag(line)
    tar, gen = BertM(bertmodel, berttoken, line)
    for sen in gen:
        f.write(tar.strip() + "\n")
        f.write(sen.strip() + "\n")
f.close()
