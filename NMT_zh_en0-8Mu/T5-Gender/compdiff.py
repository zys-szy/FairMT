import os

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

with open("./en_mu.txt") as f:
    lines = f.readlines()

with open("./f_en_mu.zh.beam") as f:
    zhlines = f.readlines()
    zhlines = [line.split("\t")[0] for line in zhlines]

f = open("./diff.txt", "w")
for i in range(0, len(lines), 2):
    f.write(lines[i])
    f.write(zhlines[i] + "\n")
    f.write(lines[i + 1])
    f.write(zhlines[i + 1] + "\n")
    f.write(read_from_wdiff(lines[i].strip(), lines[i + 1].strip()) + "\n")
    f.write(read_from_wdiff(zhlines[i].strip(), zhlines[i + 1].strip()) + "\n\n")
    

f.close()
