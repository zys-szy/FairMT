import sys
import os
import math
import collections

def tobool(s):
    if s == "True":
        return True
    else:
        return False

filter_list = []
#with open("../../NMT_zh_en0/google/human_google/train.txt") as f:
#    lines = f.readlines()
#    f.close()
#    now_count_all = 0
#    for i in range(0, len(lines), 11):
#        now_count_all += 1
#        if now_count_all == 45:
#            continue
#        filter_list.append(lines[i + 6].strip())
#        if now_count_all >= 100:
#            break

t_out_list = [[] for t in range(5)]

ground_new = [{} for i in range(10)]

sum = [0] * 9

bugs = []
good = []

filter_count = 0
with open("Com_ALL.txt", "r") as f:
    lines = f.readlines()
    count_data = len(lines) // 15
    for i in range(0, len(lines), 15):
        vote = 0
        if lines[i + 10].strip() in filter_list:
            filter_count += 1
            continue
        for t in range(9):
            #ground_new[t][lines[i + 12].strip()] = (tobool(lines[i + t].strip().split()[-1]))
            length = 1#float(max(len(lines[i + 9].split()), len(lines[i + 11].split())))
            #length = math.sqrt(length)
            if t < 4:
                #continue
                delta = float(lines[i + t].strip().split()[1])
                sc1 = float(lines[i + t].strip().split()[2])
                sc2 = float(lines[i + t].strip().split()[3])
                if t != 3:
                    delta *= length
                    sc1 *= length
                    sc2 *= length
                #length = len()
                if delta >= max(sc1, sc2) * [0.05, 0.08, 0.06, 0.08][t]:
                    ground_new[t][lines[i + 10].strip()] = True#(tobool(lines[i + t].strip().split()[-1]))
                else:
                    ground_new[t][lines[i + 10].strip()] = False
                if ground_new[t][lines[i + 10].strip()] == False:
                    vote += 0.5
                sum[t] += float(delta)
            else:
                #sum[t] += 1 - float(lines[i + t].strip().split()[1])
                if t == 7:
                    score = float(lines[i + t].strip().split()[1])
                    if float(lines[i + t].strip().split()[1]) < 0.906:
                        ground_new[t][lines[i + 10].strip()] = True
                    else:
                        ground_new[t][lines[i + 10].strip()] = False
                        vote += 1
                else:
                    score = float(lines[i + t].strip().split()[1]) * length
                    if score < [0.963, 0.963, 0.999, 0, 0][t - 4]:
                        if t - 4 == 0:
                            bugs.append([lines[i + 12].strip(), lines[i + 11].strip(), lines[i + 10].strip(), lines[i + 9].strip()])
                        ground_new[t][lines[i + 10].strip()] = True
                    else:
                        if t == 4:
                            good.append([lines[i + 12].strip(), lines[i + 11].strip(), lines[i + 10].strip(), lines[i + 9].strip()])
                        ground_new[t][lines[i + 10].strip()] = False
                        vote += 1
                sum[t] += 1 - score

        if vote >= 3:
            ground_new[8][lines[i + 10].strip()] = False
        else:
            ground_new[8][lines[i + 10].strip()] = True


import random
import difflib
random.shuffle(bugs)
random.shuffle(good)

def wdiff (s1, s2):
    with open("m1", "w") as f:
        f.write(s1)
    with open("m2", "w") as f:
        f.write(s2)
    diff = os.popen(f"wdiff m1 m2")
    return diff.readline() + "\n"

f = open("target.txt", "w")
for i in range(200):
    data = bugs[i]
    for k in data:
        f.write(k + "\n")
    diff = wdiff(data[0],data[2])
    f.write(diff)#' '.join(list(diff)).replace("\n", " ") + "\n")
    diff = wdiff(data[1],data[3])
    f.write(diff)#' '.join(list(diff)).replace("\n", " ") + "\n")
    f.write("-----\n")
    
    data = good[i]
    for k in data:
        f.write(k + "\n")
    diff = wdiff(data[0],data[2])
    f.write(diff)#' '.join(list(diff)).replace("\n", " ") + "\n")
    diff = wdiff(data[1],data[3])
    f.write(diff)#' '.join(list(diff)).replace("\n", " ") + "\n")
    f.write("-----\n")
f.close()



