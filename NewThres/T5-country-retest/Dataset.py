import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
import h5py
from tqdm import tqdm
import json
import jieba
from copy import deepcopy
from scipy import sparse
def splitCamel(s):
    if s.isupper():
        return [s.lower()]
    ans = []
    tmpans = ""
    for x in s:
        if x.isupper() or x == '_':
            if tmpans != "":
                ans.append(tmpans)
            tmpans = x.replace("_", "")
        else:
            tmpans += x
    if tmpans != "":
        ans.append(tmpans)
    for i in range(len(ans)):
        ans[i] = ans[i].lower()
    return ans
def isnum(str_number):
    return (str_number.split(".")[0]).isdigit() or str_number.isdigit() or  (str_number.split('-')[-1]).split(".")[-1].isdigit()
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train"):
        self.train_path = "train.txt"
        self.val_path = "dev.txt"  # "validD.txt"
        self.test_path = "test.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Table_Len = config.TableLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50
        self.ruledict = pickle.load(open("rule.pkl", "rb"))
        #self.ruledict["start -> root"] = len(self.ruledict)
        #self.ruledict["start -> selecLocVar"] = len(self.ruledict)
        #self.ruledict["start -> selectMethod"] = len(self.ruledict)
        #self.ruledict["start -> selecVar"] = len(self.ruledict)
        self.rrdict = {}
        #self.tables = json.load(open("tables2.json", "r"))
        self.tablename = {}
        self.cache = {}
        self.typedict = {'pad':0}
        self.edgedict = {'pad':0}
        for x in self.ruledict:
            self.rrdict[self.ruledict[x]] = x
        if not os.path.exists("nl_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        #print(self.Nl_Voc)
        if dataName == "train":
            if os.path.exists("data.pkl"):
                self.data = pickle.load(open("data.pkl", "rb"))
                self.nl = pickle.load(open("nl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.train_path, "r", encoding='utf-8'))
        elif dataName == "val":
            if os.path.exists("valdata.pkl"):
                self.data = pickle.load(open("valdata.pkl", "rb"))
                self.nl = pickle.load(open("valnl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.val_path, "r", encoding='utf-8'))
        elif dataName == "eval":
            if os.path.exists("evaldata.pkl"):
                self.data = pickle.load(open("evaldata.pkl", "rb"))
                self.nl = pickle.load(open("evalnl.pkl", "rb"))
                self.tabless = pickle.load(open("evaltable.pkl", "rb"))
                return
            self.data = self.preProcessDataEval(open('eval.txt', "r", encoding='utf-8'))
        else:
            if os.path.exists("testdata.pkl"):
                self.data = pickle.load(open("testdata.pkl", "rb"))
                self.tabless = pickle.load(open("testtable.pkl", "rb"))
                self.nl = pickle.load(open("testnl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.test_path, "r", encoding='utf-8'))
        if dataName == 'test':
            print(self.edgedict)

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
        if os.path.exists("typedict.pkl"):
            self.typedict = pickle.load(open("typedict.pkl", "rb"))
        if os.path.exists("edgedict.pkl"):
            self.edgedict = pickle.load(open("edgedict.pkl", "rb"))
        #self.Nl_Voc["<emptynode>"] = len(self.Nl_Voc)
        #self.Code_Voc["<emptynode>"] = len(self.Code_Voc)

    def init_dic(self):
        print("initVoc")
        f = open(self.train_path, "r", encoding='utf-8')
        lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        nls = []
        rules = []
        for i in tqdm(range(int(len(lines) / 8))):
            data = lines[8 * i].strip().lower().split()
            nls.append(data)
            var = eval(lines[8 * i + 5].strip())
            for x in var:
                rules.append(splitCamel(x))
                #rules.append([var[x]])
            method = eval(lines[8 * i + 6].strip())
            for x in method:
                rules.append(splitCamel(x))
            types = eval(lines[8 * i + 7].strip())
            for x in types:
                rules.append(splitCamel(x))
        rules.append(['method_type', 'var_type', 'nlnumber', 'nltext', 'type'])
        f.close()
        nl_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=0)
        code_voc = VocabEntry.from_corpus(rules, size=50000, freq_cutoff=0)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id
        for x in self.ruledict:
            lst = x.strip().lower().split()
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.Code_Voc:
                    self.Code_Voc[y] = len(self.Code_Voc)
            #rules.append([lst[0]] + lst[2:])
        #print(self.Code_Voc)
        assert("root" in self.Code_Voc)
        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
        print(maxNlLen, maxCodeLen, maxCharLen)
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def gen_next(self, s2):
        k = -1
        n = len(s2)
        j = 0
        next_list = [0 for i in range(n)]
        next_list[0] = -1                           #next数组初始值为-1
        while j < n-1:
            if k == -1 or s2[k] == s2[j]:
                k += 1
                j += 1
                next_list[j] = k                    #如果相等 则next[j+1] = k
            else:
                k = next_list[k]                    #如果不等，则将next[k]的值给k
        return next_list


    def match(self, s1, s2):
        sc = s1 + "kk" + s2
        if sc in self.cache:
            #sc = s1 + "kk" + s2
            return self.cache[sc] / len(s2)
        #elif len(s2) >= len(s1) and s2 + "kk" + s1 in self.cache:
        #    sc = s2 + "kk" + s1
        #    return self.cache[s2 + "kk" + s1] / len(s2)
        #if len(s2) < len(s1):
        #    sc = s1 + "kk" + s2
        #else:
        #    sc = s2 + "kk" + s1
        ans = -1
        next_list = self.gen_next(s2)
        i = 0
        j = 0
        ma_len = 0
        while i < len(s1):
            if s1[i] == s2[j] or j == -1:
                i += 1
                j += 1
                ma_len = max(ma_len, j)
            else:
                j = next_list[j]
            if j == len(s2):
                ans = i - len(s2)
                break
        self.cache[sc] = ma_len
        return ma_len / len(s2)
    def preProcessData(self, dataFile):
        lines = dataFile.readlines()
        inputNl = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        inputParentList = []
        inputTable = []
        inputTabelChar = []
        inputTablead = []
        inputTabletype = []
        nls = []
        ttables = []
        inputNlType = []
        for i in tqdm(range(int(len(lines) / 8))):
            tableadrow = []#np.zeros([self.Table_Len, self.Table_Len])
            tableadcol = []
            tableaddata = []
            tmp = lines[8 * i + 1].strip().split()
            outr = False
            for x in tmp:
                x = int(x)
                if x - 30000 >= self.Table_Len and x <= 2000000:
                    outr = True
                    break
            if outr:
                continue
            child = {}
            nl = lines[8 * i].lower().strip().split( )
            nltype = []
            for t, x in enumerate(nl):
                if isnum(x):
                    #if 'nlnumber' not in self.typedict:
                    #    self.typedict['nlnumber'] = len(self.typedict)
                    nltype.append(int(self.Code_Voc['nlnumber']))
                else:
                    #if 'nltext' not in self.typedict:
                    #    self.typedict['nltext'] = len(self.typedict)
                    nltype.append(int(self.Code_Voc['nltext']))
            nltype = self.pad_seq(nltype, self.Nl_Len)
            inputNlType.append(nltype)
            nls.append(nl)
            inputparent = lines[8 * i + 2].strip().split()
            inputres = lines[8 * i + 1].strip().split()
            parentname = lines[8 * i + 4].strip().lower().split()
            methods = eval(lines[8 * i + 6].strip())
            varss = eval(lines[8 * i + 5].strip())
            vartype = eval(lines[8 * i + 7].strip())
            inputadrow = []#np.zeros([self.Table_Len, self.Table_Len])
            inputadcol = []
            inputaddata = []#[]#np.zeros([self.Nl_Len + self.Table_Len + self.Code_Len, self.Nl_Len + self.Table_Len + self.Code_Len])
            #for i in range(self.Nl_Len + self.Table_Len):
            #    inputad.append([self.Nl_Len + self.Table_Len + self.Code_Len + 1])
            #inputad.append([self.Nl_Len + self.Table_Len + self.Code_Len + 1])
            inputrule = [self.ruledict["start -> root"]]
            for j in range(len(inputres)):
                inputres[j] = int(inputres[j])
                inputparent[j] = int(inputparent[j]) + 1
                assert(inputres[j] != 0)
                if inputres[j] >= 2000000:
                    assert(0)
                    inputres[j] = len(self.ruledict) + inputres[j] - 2000000 + 1
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Table_Len + j + 1)
                        inputadcol.append(self.Table_Len + inputres[j] - len(self.ruledict))
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + self.Table_Len + j + 1].append(inputres[j] - len(self.ruledict))
                        #inputad[self.Nl_Len + self.Table_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> selecLocVar'])
                elif inputres[j] >= 20000 and inputres[j] < 30000:
                    print(inputres)
                    assert(0)
                    inputres[j] = len(self.ruledict) + self.Nl_Len + inputres[j] - 30000 
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + self.Table_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict))
                        #print('1', inputres[j] - len(self.ruledict))
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + self.Table_Len + j + 1].append(inputres[j] - len(self.ruledict))
                        #inputad[self.Nl_Len + self.Table_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> copyword'])
                elif inputres[j] >= 30000:
                    inputres[j] = len(self.ruledict) + inputres[j] - 30000
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Table_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict))
                        inputaddata.append(1)
                        #print('1', inputres[j] - len(self.ruledict))
                        #inputad[self.Nl_Len + self.Table_Len + j + 1].append(inputres[j] - len(self.ruledict))
                        #inputad[self.Nl_Len + self.Table_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    if inputres[j] - len(self.ruledict) - self.Code_Len < len(methods):
                        inputrule.append(self.ruledict['start -> selectMethod'])
                    elif inputres[j] - len(self.ruledict) - self.Code_Len < len(methods) + len(varss):
                        inputrule.append(self.ruledict['start -> selecVar'])
                    elif inputres[j] - len(self.ruledict) - self.Code_Len < len(methods) + len(varss) +len(vartype):
                        inputrule.append(self.ruledict['start -> selectype'])
                    else:
                        print(parentname[j])
                        assert(0)
                else:
                    inputrule.append(inputres[j])
                child.setdefault(inputparent[j], []).append(j + 1)
                if j + 1 < self.Code_Len:
                    inputadrow.append(self.Table_Len + j + 1)
                    inputadcol.append(self.Table_Len + inputparent[j])
                    inputaddata.append(1)
                    #inputad[self.Nl_Len + self.Table_Len + j + 1].append(self.Nl_Len + self.Table_Len + inputparent[j])
                    #inputad[self.Nl_Len + self.Table_Len + j + 1, self.Nl_Len + self.Table_Len + inputparent[j]] = 1
            inputnls = []
            #for x in nl:
            #    tmp = self.pad_seq(self.Get_Em(x.split(' '), self.Nl_Voc), 10)
            #    inputnls.append(tmp)
            inputnls = self.pad_seq(self.Get_Em(nl, self.Nl_Voc), self.Nl_Len)#self.pad_list(inputnls, self.Nl_Len, 10)#self.Get_Em(nl, self.Nl_Voc)
            inputNl.append(inputnls)
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            inputruleparent = self.pad_seq(self.Get_Em(["start"] + parentname, self.Code_Voc), self.Code_Len)
            inputrulechild = []
            for x in inputrule:
                if x >= len(self.rrdict):
                    inputrulechild.append(self.pad_seq(self.Get_Em(["copyword"], self.Code_Voc), self.Char_Len))
                else:
                    rule = self.rrdict[x].strip().lower().split()
                    inputrulechild.append(self.pad_seq(self.Get_Em(rule[2:], self.Code_Voc), self.Char_Len))
            #depth = [self.pad_seq([1], 40)]

            inputparentpath = []
            for j in range(len(inputres)):
                if inputres[j] in self.rrdict:
                    tmppath = [self.rrdict[inputres[j]].strip().lower().split()[0]]
                    #print(tmppath[0], parentname[j].lower())
                    if tmppath[0] == 'start':
                        tmppath[0] = parentname[j].lower()
                    assert(tmppath[0] == parentname[j].lower())
                else:
                    tmppath = [parentname[j].lower()]
                curr = inputparent[j]
                while curr != 0:
                    rule = self.rrdict[inputres[curr - 1]].strip().lower().split()[0]
                    tmppath.append(rule)
                    curr = inputparent[curr - 1]
                inputparentpath.append(self.pad_seq(self.Get_Em(tmppath, self.Code_Voc), 10))
            inputrule = self.pad_seq(inputrule, self.Code_Len)
            inputres = self.pad_seq(inputres, self.Code_Len)
            tmp = [self.pad_seq(self.Get_Em(['start'], self.Code_Voc), 10)] + inputparentpath
            inputrulechild = self.pad_list(tmp, self.Code_Len, 10)
            inputRuleParent.append(inputruleparent)
            inputRuleChild.append(inputrulechild)
            inputRes.append(inputres)
            inputRule.append(inputrule)
            #print(inputadrow)
            #print(inputadcol)
            inputad = sparse.coo_matrix((inputaddata, (inputadrow, inputadcol)), shape=(self.Table_Len + self.Code_Len, self.Table_Len + self.Code_Len))
            inputParent.append(inputad)
            inputParentPath.append(self.pad_list(inputparentpath, self.Code_Len, 10))
            inputParentList.append(self.pad_seq(inputparent, self.Code_Len))
            #inputDepth.append(depth)
            inputtable = []
            tables = []
            tabletype = []
            tokenid = {}
            for x in methods:
                tokenid[x + 'method'] = len(tables)
                tmp = splitCamel(x)
                inputtable.append(self.pad_seq(self.Get_Em(tmp, self.Code_Voc), 10))
                tables.append(x.lower())
                tabletype.append(self.Code_Voc['method_type'])
                #tmp = ['method_type']
                #for y in methods[x][0]:
                #    k = y.split()
                #    tmp.append(k[0])
                #tabletype.append(self.pad_seq(self.Get_Em(tmp, self.Code_Voc), 10))
            for x in varss:
                tokenid[x + 'var'] = len(tables)
                tmp = splitCamel(x)
                inputtable.append(self.pad_seq(self.Get_Em(tmp, self.Code_Voc), 10))
                tables.append(x.lower())
                tabletype.append(self.Code_Voc['var_type'])
                #tmp = ['var_type']
                #tmp.append(varss[x])
                #tabletype.append(self.pad_seq(self.Get_Em(tmp, self.Code_Voc), 10))
            for x in vartype:
                tokenid[x + 'type'] = len(tables)
                tmp = splitCamel(x)
                inputtable.append(self.pad_seq(self.Get_Em(tmp, self.Code_Voc), 10))
                tables.append(x.lower())
                tabletype.append(self.Code_Voc['type'])
            ttype = []
            inputtable = self.pad_list(inputtable, self.Table_Len, 10)
            tabletype = self.pad_seq(tabletype, self.Table_Len)
            #make edge
            for x in varss:
                id1 = self.Nl_Len + tokenid[x + 'var']
                id2 = self.Nl_Len + tokenid[varss[x] + 'type']
                if id1 >= self.Nl_Len + self.Table_Len or id2 >= self.Nl_Len + self.Table_Len:
                    continue
                tableadrow.append(id1)
                tableadcol.append(id2)
                if 'typeof' not in self.edgedict:
                    self.edgedict['typeof'] = len(self.edgedict)
                tableaddata.append(self.edgedict['typeof'])
                tableadrow.append(id2)
                tableadcol.append(id1)
                if 'typeis' not in self.edgedict:
                    self.edgedict['typeis'] = len(self.edgedict)
                tableaddata.append(self.edgedict['typeis'])
            for x in methods:
                for l, y in enumerate(methods[x][0]):
                    y = y.split()[0]
                    id1 = self.Nl_Len + tokenid[x + 'method']
                    id2 = self.Nl_Len + tokenid[y + 'type']
                    if id1 >= self.Nl_Len + self.Table_Len or id2 >= self.Nl_Len + self.Table_Len:
                        continue
                    if l == 0:
                        tableadrow.append(id1)
                        tableadcol.append(id2)
                        if 'rettypeof' not in self.edgedict:
                            self.edgedict['rettypeof'] = len(self.edgedict)
                        tableaddata.append(self.edgedict['rettypeof'])
                        tableadrow.append(id2)
                        tableadcol.append(id1)
                        if 'rettypeis' not in self.edgedict:
                            self.edgedict['rettypeis'] = len(self.edgedict)
                        tableaddata.append(self.edgedict['rettypeis'])
                    else:
                        tableadrow.append(id1)
                        tableadcol.append(id2)
                        if 'argtypeof' + str(l) not in self.edgedict and self.dataName == 'train':
                            self.edgedict['argtypeof' + str(l)] = len(self.edgedict)
                        if 'argtypeof' + str(l) not in self.edgedict:
                            tableaddata.append(self.edgedict['argtypeof' + str(1)])
                        else:
                            tableaddata.append(self.edgedict['argtypeof' + str(l)])
                        #tableadrow.append(id2)
                        #tableadcol.append(id1)
                        #if 'rettypeis' not in self.edgedict:
                        #    self.edgedict['rettypeis'] = len(self.edgedict)
                        #tableaddata.append(self.edgedict['rettypeis'])
            inputTabletype.append(tabletype)
            tablead = (tableadrow, tableadcol, tableaddata)#sparse.coo_matrix((tableaddata, (tableadrow, tableadcol)), shape=(self.Nl_Len + self.Table_Len, self.Nl_Len + self.Table_Len))
            tdic = {}
            #for k in range(len(tableadrow)):
            #    if (tableadrow[k], tableadcol[k]) in tdic:
            #        print(i, j, tableaddata[k])
            #        print(methods)
            #        print(varss)
            #        print(vartype)
            #        print(self.edgedict)
            #        assert(0)
            #    tdic[(tableadrow[k], tableadcol[k])] = 1
            inputtablechar = self.Get_Char_Em(tables)
            for j in range(len(inputtablechar)):
                inputtablechar[j] = self.pad_seq(inputtablechar[j], self.Char_Len)
            inputtablechar = self.pad_list(inputtablechar, self.Table_Len, self.Char_Len)
            inputTabelChar.append(inputtablechar)
            inputTablead.append(tablead)
            inputTable.append(inputtable)
            #print(tables)
            ttables.append(tables)
            for x in inputres:
                #print(nl)
                assert((x - len(self.ruledict)) < self.Nl_Len + len(tables))
        batchs = [inputNl, inputNlType, inputRule, inputRuleParent, inputRuleChild, inputRes, inputParent, inputParentPath, inputTable, inputTablead, inputTabletype, inputNlChar, inputTabelChar]
        self.data = batchs
        self.nl = nls
        self.tables = tables
        #self.code = codes
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("nl.pkl", "wb").write(pickle.dumps(nls))
            open("table.pkl", "wb").write(pickle.dumps(self.tables))
            #open('typedict.pkl', 'wb').write(pickle.dumps(self.typedict))
            open('edgedict.pkl', 'wb').write(pickle.dumps(self.edgedict))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("valnl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "test":
            open("testdata.pkl", "wb").write(pickle.dumps(batchs))
            #open("testcode.pkl", "wb").write(pickle.dumps(self.code))
            open("testnl.pkl", "wb").write(pickle.dumps(self.nl))
            open("testtable.pkl", "wb").write(pickle.dumps(self.tables))
        return batchs
    def preProcessDataEval(self, dataFile):
        lines = dataFile.readlines()
        inputNl = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        inputParentList = []
        inputTable = []
        inputTabelChar = []
        inputTablead = []
        inputTabletype = []
        nls = []
        ttables = []
        inputNlType = []
        for i in tqdm(range(int(len(lines) / 2))):
            child = {}
            nl = lines[2 * i].lower().strip().split()
            nltype = []
            nlsegid = []
            for k, x in enumerate(nl):
                if x == 'seg':
                    nlsegid.append(k)
                if isnum(x):
                    if 'nlnumber' not in self.typedict:
                        self.typedict['nlnumber'] = len(self.typedict)
                    nltype.append(self.typedict['nlnumber'])
                else:
                    if 'nltext' not in self.typedict:
                        self.typedict['nltext'] = len(self.typedict)
                    nltype.append(self.typedict['nltext'])
            nltype = self.pad_seq(nltype, self.Nl_Len)
            inputNlType.append(nltype)
            nls.append(nl)
            dbid = lines[2 * i + 1].strip().lower()
            inputnls = []
            for x in nl:
                tmp = self.pad_seq(self.Get_Em(jieba.lcut(x), self.Nl_Voc), 10)
                inputnls.append(tmp)
            inputnls = self.pad_list(inputnls, self.Nl_Len, 10)#self.Get_Em(nl, self.Nl_Voc)
            inputNl.append(inputnls)
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            inputtable = []
            tables = []
            tabletype = []
            tableadrow = []#np.zeros([self.Table_Len, self.Table_Len])
            tableadcol = []
            tableaddata = []
            ttype = []
            for x in [self.tablename[dbid]]:
                colid = {}
                for p, y in enumerate(x['table_names']):
                    tmp = jieba.lcut(y.lower())
                    inputtable.append(self.pad_seq(self.Get_Em(tmp, self.Nl_Voc), 10))
                    tables.append(y)
                    if 'table' not in self.typedict:
                        self.typedict['table'] = len(self.typedict)
                    tabletype.append(self.typedict['table'])
                    ttype.append('table')
                    if 'tableiden' not in self.edgedict:
                        self.edgedict['tableiden'] = len(self.edgedict)
                    tableadrow.append(self.Nl_Len + p)
                    tableadcol.append(self.Nl_Len + p)
                    tableaddata.append(self.edgedict['tableiden'])
                for j, y in enumerate(x['column_names']):
                    if y[0] == -1:
                        y[0] = 0
                    colid[j] = len(tabletype)
                    tableadrow.append(self.Nl_Len + y[0])
                    tableadcol.append(self.Nl_Len + len(tabletype))
                    if 'hascolumn' not in self.edgedict:
                        self.edgedict['hascolumn'] = len(self.edgedict)
                    tableaddata.append(self.edgedict['hascolumn'])
                    tableadrow.append(self.Nl_Len + len(tabletype))
                    tableadcol.append(self.Nl_Len + y[0])
                    if 'belongsto' not in self.edgedict:
                        self.edgedict['belongsto'] = len(self.edgedict)
                    tableaddata.append(self.edgedict['belongsto'])
                    tables.append(y[1])
                    colname = y[1].replace(" ( ", '🍬').replace("( ", '🍬').replace(' ) ', ' 🔥 ').replace(' )', ' 🔥 ').replace("(", "_").replace(")", "_").replace( '🍬', " ( ").replace(' 🔥 ', ' ) ').replace('1~10', '10').replace('~', '_').replace(":", "").replace(' / ', '✨').replace('/', '_').replace('✨', ' / ').replace(" ", "")
                    tmp = jieba.lcut(colname.lower())  
                    inputtable.append(self.pad_seq(self.Get_Em(tmp, self.Nl_Voc), 10))
                    colidx = len(tabletype)
                    tableadrow.append(self.Nl_Len + colidx)
                    tableadcol.append(self.Nl_Len + colidx)
                    if 'coliden' not in self.edgedict:
                        self.edgedict['coliden'] = len(self.edgedict)
                    tableaddata.append(self.edgedict['coliden'])
                    if 'col' + x['column_types'][j] not in self.typedict:
                        self.typedict['col' + x['column_types'][j]] = len(self.typedict)
                    tabletype.append(self.typedict['col' + x['column_types'][j]])
                    ttype.append('column')
                for j, y in enumerate(x['column_names']):
                    if y[1] == '*':
                        continue
                    continue
                    colidx = colid[j]
                    if j != 0:
                        content = tablecontentr[dbid]['tables'][x['table_names'][y[0]]]['cell']
                        for k in range(len(tablecontentr[dbid]['tables'][x['table_names'][y[0]]]['header'])):
                            if tablecontentr[dbid]['tables'][x['table_names'][y[0]]]['header'][k] == y[1]:
                                break
                        for m in range(len(content)):
                            s = content[m][k]
                            if s == '':
                                s = 'empty'
                            s = s.replace(" ( ", '🍬').replace("( ", '🍬').replace(' ) ', ' 🔥 ').replace(' )', ' 🔥 ').replace("(", "_").replace(")", "_").replace( '🍬', " ( ").replace(' 🔥 ', ' ) ').replace('1~10', '10').replace('~', '_').replace(":", "").replace(' / ', '✨').replace('/', '_').replace('✨', ' / ').lower().replace(" ", "")
                            '''zhmodel = re.compile('[\u4e00-\u9fa5]')
                            res = zhmodel.search(s)
                            if not res:
                                if s == '1亿':
                                    assert(0)
                                continue'''
                            if content[m][k] == '':
                                tables.append('empty')
                            else:
                                tables.append(content[m][k])
                            #assert(s != 'item_geostatistics_4_19')
                            #tablead.append([])
                            tmp = jieba.lcut(s.lower()) 
                            inputtable.append(self.pad_seq(self.Get_Em(tmp, self.Nl_Voc), 10))
                            tableadrow.append(self.Nl_Len + len(tabletype))
                            tableadcol.append(self.Nl_Len + colidx)
                            if 'cellbelongto' not in self.edgedict:
                                self.edgedict['cellbelongto'] = len(self.edgedict)
                            tableaddata.append(self.edgedict['cellbelongto'])
                            tableadrow.append(self.Nl_Len + colidx)
                            tableadcol.append(self.Nl_Len + len(tabletype))
                            if 'columnhascell' not in self.edgedict:
                                self.edgedict['columnhascell'] = len(self.edgedict)
                            tableaddata.append(self.edgedict['columnhascell'])
                            tableadrow.append(self.Nl_Len + len(tabletype))
                            tableadcol.append(self.Nl_Len + len(tabletype))
                            if 'celliden' not in self.edgedict:
                                self.edgedict['celliden'] = len(self.edgedict)
                            tableaddata.append(self.edgedict['celliden'])
                            if 'cell' + x['column_types'][j] not in self.typedict:
                                self.typedict['cell' + x['column_types'][j]] = len(self.typedict)
                            tabletype.append(self.typedict['cell' + x['column_types'][j]])  
                            ttype.append('cell')
                for j, y in enumerate(x['column_names']):
                    for k, z in enumerate(x['column_names']):
                        if j != k and y[0] == z[0]:
                            tableadrow.append(self.Nl_Len + colid[j])
                            tableadcol.append(self.Nl_Len + colid[k])
                            if 'same_table' not in self.edgedict:
                                self.edgedict['same_table'] = len(self.edgedict)
                            tableaddata.append(self.edgedict['same_table']) 
                            tableadrow.append(self.Nl_Len + colid[k])
                            tableadcol.append(self.Nl_Len + colid[j])
                            if 'same_table' not in self.edgedict:
                                self.edgedict['same_table'] = len(self.edgedict)
                            tableaddata.append(self.edgedict['same_table']) 
                for y in x['primary_keys']:
                    tableid = x['column_names'][y][0]
                    for j in range(len(tableadrow)):
                        if tableadrow[j] == self.Nl_Len + tableid and tableadcol[j] == self.Nl_Len + colid[y]:
                            if 'hasprikey' not in self.edgedict:
                                self.edgedict['hasprikey'] = len(self.edgedict)
                            tableaddata[j] = self.edgedict['hasprikey']
                        if tableadcol[j] == self.Nl_Len + tableid and tableadrow[j] == self.Nl_Len + colid[y]:
                            if 'isprikeyof' not in self.edgedict:
                                self.edgedict['isprikeyof'] = len(self.edgedict)
                            tableaddata[j] = self.edgedict['isprikeyof']
                for y in x['foreign_keys']:
                    tableadrow.append(self.Nl_Len + colid[y[0]])
                    tableadcol.append(self.Nl_Len + colid[y[1]])
                    if 'coliskeyfor' not in self.edgedict:
                        self.edgedict['coliskeyfor'] = len(self.edgedict)
                    tableaddata.append(self.edgedict['coliskeyfor'])
                    tableadrow.append(self.Nl_Len + colid[y[1]])
                    tableadcol.append(self.Nl_Len + colid[y[0]])
                    if 'coliskeyof' not in self.edgedict:
                        self.edgedict['coliskeyof'] = len(self.edgedict)
                    tableaddata.append(self.edgedict['coliskeyof'])
                    tableid1 = x['column_names'][y[0]][0]
                    tableid2 = x['column_names'][y[1]][0]
                    hasrel = False
                    for j in range(len(tableadrow)):
                        if tableadrow[j] == self.Nl_Len + tableid2 and tableadcol[j] == self.Nl_Len + tableid1 and tableaddata[j] == self.edgedict['forkeyof']:
                            if 'forkeyboth' not in self.edgedict:
                                self.edgedict['forkeyboth'] = len(self.edgedict)
                            tableaddata[j] = self.edgedict['forkeyboth']
                            for k in range(len(tableadrow)):
                                if tableadrow[k] == self.Nl_Len + tableid1 and tableadcol[k] == self.Nl_Len + tableid2 and tableaddata[k] == self.edgedict['forkeyfor']:
                                    tableaddata[k] = self.edgedict['forkeyboth']
                            hasrel = True
                        if tableadrow[j] == self.Nl_Len + tableid2 and tableadcol[j] == self.Nl_Len + tableid1 and tableaddata[j] == self.edgedict['forkeyfor']:
                            hasrel = True
                    if not hasrel:
                        tableadrow.append(self.Nl_Len + tableid1)
                        tableadcol.append(self.Nl_Len + tableid2)
                        if 'forkeyof' not in self.edgedict:
                            self.edgedict['forkeyof'] = len(self.edgedict)
                        tableaddata.append(self.edgedict['forkeyof'])
                        tableadrow.append(self.Nl_Len + tableid2)
                        tableadcol.append(self.Nl_Len + tableid1)
                        if 'forkeyfor' not in self.edgedict:
                            self.edgedict['forkeyfor'] = len(self.edgedict)
                        tableaddata.append(self.edgedict['forkeyfor'])
            inputtable = self.pad_list(inputtable, self.Table_Len, 10)
            tabletype = self.pad_seq(tabletype, self.Table_Len)
            inputTabletype.append(tabletype)
            for p in range(len(nl)):
                if  p >= self.Nl_Len:
                    break
                for k in range(len(nl)):
                    if k >= self.Nl_Len:
                        break
                    d = max(-2, min(2, k - p))
                    edgename = 'nl' + str(d)
                    tableadrow.append(p)
                    tableadcol.append(k)
                    if edgename not in self.edgedict:
                        self.edgedict[edgename] = len(self.edgedict)
                    tableaddata.append(self.edgedict[edgename])
            for p in range(len(nl)):
                if  p >= self.Nl_Len:
                    break
                for q in range(len(tables)):
                    if nl[p] == tables[q]:
                        #print(p, q)
                        edgename = 'nlhas' + str(ttype[q])
                        tableadrow.append(p)
                        tableadcol.append(self.Nl_Len + q)
                        if edgename not in self.edgedict:
                            self.edgedict[edgename] = len(self.edgedict)
                        tableaddata.append(self.edgedict[edgename])
                        tableadrow.append(self.Nl_Len + q)
                        tableadcol.append(p)
                        edgename = str(ttype[q]) + "hasnl"#$+ "_" + str(int(self.match(nl[p], tables[q]) * 10))
                        if edgename not in self.edgedict:
                            self.edgedict[edgename] = len(self.edgedict)
                        tableaddata.append(self.edgedict[edgename])
                    elif self.match(nl[p], tables[q]):
                        edgename = 'nlpart' + str(ttype[q])
                        tableadrow.append(p)
                        tableadcol.append(self.Nl_Len + q)
                        if edgename not in self.edgedict:
                            self.edgedict[edgename] = len(self.edgedict)
                        tableaddata.append(self.edgedict[edgename])
                        tableadrow.append(self.Nl_Len + q)
                        tableadcol.append(p)
                        edgename = str(ttype[q]) + "partnl"
                        if edgename not in self.edgedict:
                            self.edgedict[edgename] = len(self.edgedict)
                        tableaddata.append(self.edgedict[edgename])
                    else:
                        edgename = 'nlno' + str(ttype[q])
                        tableadrow.append(p)
                        tableadcol.append(self.Nl_Len + q)
                        if edgename not in self.edgedict:
                            self.edgedict[edgename] = len(self.edgedict)
                        tableaddata.append(self.edgedict[edgename])
                        tableadrow.append(self.Nl_Len + q)
                        tableadcol.append(p)
                        edgename = str(ttype[q]) + "nonl"
                        if edgename not in self.edgedict:
                            self.edgedict[edgename] = len(self.edgedict)
                        tableaddata.append(self.edgedict[edgename])
            tablead = sparse.coo_matrix((tableaddata, (tableadrow, tableadcol)), shape=(self.Nl_Len + self.Table_Len, self.Nl_Len + self.Table_Len))
            '''tdic = {}
            for k in range(len(tableadrow)):
                if (tableadrow[k], tableadcol[k]) in tdic:
                    print(i, j, tableaddata[k])
                    assert(0)
                tdic[(tableadrow[k], tableadcol[k])] = 1'''
            inputtablechar = self.Get_Char_Em(tables)
            for j in range(len(inputtablechar)):
                inputtablechar[j] = self.pad_seq(inputtablechar[j], self.Char_Len)
            inputtablechar = self.pad_list(inputtablechar, self.Table_Len, self.Char_Len)
            inputTabelChar.append(inputtablechar)
            inputTablead.append(tablead)
            inputTable.append(inputtable)
            #print(tables)
            ttables.append(tables)
        batchs = [inputNl, inputNlType, inputTable, inputTablead, inputTabletype, inputNlChar, inputTabelChar]
        self.data = batchs
        self.nl = nls
        self.tabless = ttables
        #self.code = codes
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("nl.pkl", "wb").write(pickle.dumps(nls))
            open("table.pkl", "wb").write(pickle.dumps(self.tables))
            open('typedict.pkl', 'wb').write(pickle.dumps(self.typedict))
            open('edgedict.pkl', 'wb').write(pickle.dumps(self.edgedict))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("valnl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "test":
            open("testdata.pkl", "wb").write(pickle.dumps(batchs))
            #open("testcode.pkl", "wb").write(pickle.dumps(self.code))
            open("testnl.pkl", "wb").write(pickle.dumps(self.nl))
            open("testtable.pkl", "wb").write(pickle.dumps(self.tabless))
        return batchs
    def __getitem__(self, offset):
        ans = []
        if self.dataName != 'eval':
            for i in range(len(self.data)):
                d = self.data[i][offset]
                #print(d)
                if i == 9:
                    tmp = np.zeros([self.Table_Len + self.Nl_Len, self.Table_Len + self.Nl_Len, 10])
                    ma = {}
                    for j in range(len(d[0])):
                        if (d[0][j], d[1][j]) not in ma:
                            ma[(d[0][j], d[1][j])] = 0
                        else:
                            if ma[(d[0][j], d[1][j])] < 10:
                                tmp[d[0][j], d[1][j], ma[(d[0][j], d[1][j])]] = d[2][j]
                            ma[(d[0][j], d[1][j])] += 1
                    #tmp = d.toarray()
                    ans.append(tmp)
                elif i == 6:
                    #d = np.array(d)
                    #print(d.dtype)
                    tmp = d.toarray()
                    #print(tmp)
                    #print(tmp.dtype)
                    tmp = np.array(tmp).astype(np.int64)
                    ans.append(tmp)
                    #print(tmp.dtype)
                else:
                    ans.append(np.array(d).astype(np.int64))
        else:
            for i in range(len(self.data)):
                d = self.data[i][offset]
                if i == 3:
                    #d = np.array(d)
                    #print(d.dtype)
                    tmp = d.toarray()
                    ans.append(np.array(tmp))
                else:
                    ans.append(np.array(d))
        return ans
    def __len__(self):
        return len(self.data[0])
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.sibiling = None
    
#dset = SumDataset(args)
