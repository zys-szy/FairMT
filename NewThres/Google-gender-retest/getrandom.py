with open("./f_en_mu.txt") as f:
    lines = f.readlines()

with open("./outtarget.txt") as f:
    tarlines = f.readlines()

with open("./final.txt") as f:
    finallines = f.readlines()
# tens = [0] * 5

tp = []
for i in range(0, len(lines), 2):
    tp.append([lines[i].strip().replace(" ##", ""), lines[i + 1].strip().replace(" ##", "")])

tar = []
for i in range(0, len(tarlines), 4):
    tar.append([[tarlines[i].strip(), tarlines[i + 1].strip()], finallines[(i + 1) // 4]])

import random 
random.shuffle(tp)# for i in range(5)]


#count = 0
with open("target.txt", "w") as f:
    #for key in range(len(tp)):
    if True:
        nowtp = tp
        count = 0
        for t in nowtp:
            f.write(t[0] + "\n")
            f.write(t[1] + "\n")
            start = 0
            enda = 10000
            endb = 10000
            tokens0 = t[0].split()
            tokens1 = t[1].split()#[:-1]
            for i in range(min(len(tokens0), len(tokens1))):
                if tokens0[i] != tokens1[i]:
                    start = i 
                    break 
            
            for i in range(min(len(tokens0), len(tokens1))):
                if tokens0[len(tokens0) - 1 - i] != tokens1[len(tokens1) - 1 - i]:
                    print ( tokens0[len(tokens0) - 1 - i], tokens1[len(tokens1) - 1 - i])
                    
                    enda = len(tokens0) - 1 - i 
                    endb = len(tokens1) - 1 - i 
                    break 
            f.write(" ".join(tokens0[start: enda + 1]) + " --> " + " ".join(tokens1[start: endb + 1]) + "\n")
            f.write("-----------------------------------------\n")
            count += 1
            if count == 200:
                break
