from copy import deepcopy
#from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import sys
#import _thread
import time
tttt = time.time()

count_all = 0

def tqdm(a):
    return a

nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", port=34132, lang="en")

f = open("en_tk.txt", "r")
en_lines = f.readlines()
f.close()

f = open("zh_ori_cut.txt", "r")
zh_lines = f.readlines()
f.close()

#f = open("word_alignment.txt", "r")
#wa_lines = f.readlines()
#f.close()

# mutation_name = []
name_index = {}
mutation_word_list = []

f = open("word_op.txt", "r")
lines = f.readlines()
f.close()

def read_word(words):
    ret = ""
    for word in words:
        ret += str(word) + " "
    return ret

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

def mutate_word (words, index, word_index, ori_line, ori_tag):
    ret = []
    #dic = {}
    #dic = eval(ori_line)
    mutate_list = mutation_word_list[index]
    for word in mutate_list:
        now_words = deepcopy(words)
        if now_words[word_index] <= "z" and now_words[word_index] >= "a":
            now_words[word_index] = str(word[0].lower()) + str(word[1:]) #+ "<edit>"
        else:
            now_words[word_index] = str(word[0].upper()) + str(word[1:]) #+ "<edit>"
        if not check_tree (ori_tag, read_word(now_words)):
            global count_all
            count_all += 1
            print (">>>>>>>>>>>>>>>>..")
            f_f.write(" ".join(words) + "\n")
            print (" ".join(words))
            f_f.write(read_word(now_words))
            print (read_word(now_words))
            f_f.write("\n")
            for t in ori_tag:
                f_f.write(str(t[1]) + " ")
                print (t[1], end=" ")
            f_f.write("\n")
            print ("")
            for t in nlp.pos_tag(read_word(now_words).strip()):
                print (t[1], end=" ")
                f_f.write(str(t[1]) + " ")
            f_f.write("\n\n")
            print ("")
            continue
        #break
        #now_words[word_index] += "<edit>"
        #print (read_word(now_words))
        ret.append(read_word(now_words))
        #break
    ret_num = -1
    l = []
    #print (dic)
    word = words[word_index]
    #if word.lower() in dic:
    #    l = dic[word.lower()]
    if len(l) == 1:
        ret_num = l[0]
    #print (ret)
    return ret, ret_num


for line in lines:
    words = line.strip().split()
    name_index[words[0]] = len(name_index)
    #mutation_name.append(words[0])

    l = []
    for i in range(1, len(words), 1):
        l.append(words[i])
    mutation_word_list.append(l)

f_en = open("en_mu.txt", "w")
f_zh = open("zh_mu.txt", "w")

#def check_tree(tag, line):
#    tag_now 

def do_padding(zh, ret_num, tag, word, ori, en, ori_tag):
    words = zh.split()
    if ret_num != -1:
        words[ret_num] = "<padding>"
        print (en)
        print (ori, "->" ,word)
    else:
        raise Exception("No padding here !")
    
    return read_word(words)

all_vec = []
all_finish = []
all_index = []
def do_mutation(start, end, site_num):
    print (1)
    for i in tqdm(range(start, end)):
        print (i)
        if i % 2 == 1:
            print ("-----", en_lines[i].strip())
            topic = en_lines[i - 1].strip()
        else:
            continue
        en = en_lines[i].strip()
        zh = zh_lines[i].strip()
        tag = nlp.pos_tag(en)
        #print (len(tag))
        #print (tag)
        #print (len(en.strip().split()))
        #print (en.strip())
        #print ("-------")
        # print (tag)
        words = en.strip().split()
        #f_en.write(en + "\n")
        #f_zh.write(zh + "\n")
        if "㎡" in en or "#" in en:
            continue
        #print (i)
        max_num = 5
        max_each = 1
        count = 0
        vec = []
        used = []
        for div in range(max_num):
            if count >= max_num:
                break
            for t in range(len(words)):
                #if count >= max_num:
                #    break
        #        count = 0
                try:
                    if tag[t][1] in ["NNS", "JJ", "NN", "NNP", "NNPS", "CD", "NNPS"] and words[t].lower() in name_index:
                        ret_list, ret_num = mutate_word(words, name_index[words[t].lower()], t, "", tag)
                 #       try:
                 #       s = do_padding(zh, ret_num, tag[t][1], write_line.split()[t]) + "\n"
                 #       except:
                 #           continue
                        for write_line in ret_list:
                            try:
    #                            s = do_padding(zh, ret_num, tag[t][1], write_line.split()[t], words[t].lower(), en, tag) + "\n"
                                #if len(write_line.strip()) == 0:
                                #    continue
                                if write_line in used:
                                    continue
                                used.append(write_line)
                                #write_line.append(used)
                                vec.append([write_line, "", en, zh, topic])
                                print (write_line)
                                # f_en.write(write_line + "\n")
                                # f_zh.write(s)
                                count += 1
                                #break
                                if count >= max_each:
                                    break
                            except:
                                continue
                    if count >= max_num:
                        break
                except:
                    continue
        if len(vec) > 0:
            all_vec[site_num].append(vec)
        #all_index[site_num].append(i)
    all_finish[site_num] = True


th_num = int(1)

l_max = len(en_lines)
length_l = l_max // (1)
for i in range(th_num):
    all_vec.append([])
    all_finish.append(False)
    do_mutation(length_l * i, min(length_l * (i + 1), l_max), i)
    #_thread.start_new_thread(do_mutation, (length_l * i, min(length_l * (i + 1), l_max), i))

#while False in all_finish:
#    pass

f_index = open("en_cross.index", "w")
for i in range(len(all_vec)):
    vecs = all_vec[i]
    for vec in vecs:
        flag = False
        for v in vec:
            #f_en.write(v[4] + "\n")
            f_en.write(v[2] + "\n")
            #f_en.write(v[4] + "\n")
            f_en.write(v[0] + "\n")
            
            f_zh.write(v[3].strip() + "\n")
            f_index.write(str(i) + "\n")
            flag = True
            #f_zh.write("\n")
            #f_zh.write(v[1])
            #f_zh.write(v[3] + "\n")
        #f_en.write("\n")
        #f_zh.write("\n")
f_zh.close()
f_en.close()
f_index.close()
print (count_all)
nlp.close()
exit()

f = open("en_mu.txt")
lines = f.readlines()
f.close()

f = open("en_cross.txt", "w")
f_en = open("en_cross.en", "w")
f_index = open("en_cross.index", "w")
for i in range(len(lines)):
    line = lines[i].strip()
    sub_lines = line.split("\t")
    for subline in sub_lines:
        nowline = subline.strip()
        if len(nowline) == 0:
            continue
        f.write(nowline + "\n")
        f_en.write(nowline.replace("<edit>", "") + "\n")
        f_index.write(str(i) + "\n")

f.close()
f_index.close()


print (time.time() - tttt)
