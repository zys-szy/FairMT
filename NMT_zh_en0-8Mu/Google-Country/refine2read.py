import os
import jieba 

with open("./Com_BERT.txt") as f:
    lines = f.readlines()

data = []
for i in range(0, len(lines), 7):
    data.append([lines[i + t].strip() for t in range(6)])

def read_from_wdiff(old_sentence, new_sentence):
    #print (old_sentence)
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

def read4data(d):
    l = "Score: \n"
    l += d[0].split()[0].replace("[", "").replace(",", "")
    l += "\n"
    d[1] = " ".join(jieba.cut(d[1], cut_all=False))
    d[3] = " ".join(jieba.cut(d[3], cut_all=False))
    l += "Ori:\n"
    l += d[2].strip() + "\n"
    l += d[1].strip() + "\n"
    l += "Mut:\n"
    l += d[4].strip() + "\n"
    l += d[3].strip() + "\n"
    l += "Diff:\n"
    l += read_from_wdiff(d[4].strip(), d[2].strip()) + "\n"
    l += read_from_wdiff(d[3].strip(), d[1].strip()) + "\n"
    return l

with open("DATA.txt", "w") as f:
    for d in data:
        f.write(read4data(d) + "\n")

