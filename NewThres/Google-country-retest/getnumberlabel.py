import os
from tqdm import tqdm

with open('./f_en_mu.txt') as f:
    lines = f.readlines()

data = [[lines[i + t].strip() for t in range(2)] for i in range(0, len(lines), 2)]

d = []
for da in data:
    #if "n da[0]:
        d.append(da)

data = d
import random
random.shuffle(data)

def read_from_wdiff(old_sentence, new_sentence):    #print (old_sentence)
    f = open("memory_2.txt", "w")
    f.write(old_sentence.strip())    
    f.close()
    f = open("memory_1.txt", "w")    
    f.write(new_sentence.strip())
    f.close()
    diff = os.popen("wdiff memory_1.txt memory_2.txt")
    lines = diff.readlines()    
    print (lines)
    assert len(lines) == 1
    diff.close()
    return lines[0]

l = []
number = 0 
for d in tqdm(data[:100]):
    print (d[0])
    print (d[1])
    print (read_from_wdiff(d[0], d[1]))
    a = input()
    while a != "1" and a != "0":
        a = input()
    l.append([d, a])
    number += int(a)

print (number)

with open('./labeled.txt', "w") as f:
    f.write(str(l))
