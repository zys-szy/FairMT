from tqdm import tqdm

with open("./Com_BERT.ori.txt") as f:
    lines = f.readlines()

with open("./f_en_mu.ori.txt") as f:
    orilines = f.readlines()
    orilines = [line.strip() for line in orilines]

with open("./Com_BERT.txt", "w") as f:
    for i in tqdm(range(0, len(lines), 7)):
        for t in range(7):
            line = lines[i + t].strip()
            if t not in [2, 4]:
                f.write(line + "\n")
            else:
                line = line.strip()
                #if len(line) == 0:
                flag = True
                for k in orilines:
                    kori = k
                    k = k.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "")
                    if line == k:
                        f.write(kori + "\n")
                        flag = False
                        break
                if flag:
                    f.write(line + "\n")
