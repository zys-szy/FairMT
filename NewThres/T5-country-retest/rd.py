with open("./News-Commentary.en-zh.en") as f:
    enlines = f.readlines()

with open("./News-Commentary.en-zh.zh") as f:
    zhlines = f.readlines()

lines = [[enlines[i].strip(), zhlines[i].strip()] for i in range(len(enlines))]
import random

random.shuffle(lines)

with open("./train.txt", "w") as f:
    for line in lines:
        if line[0].strip() == "":
            continue
        f.write(line[0] + "\n")
        f.write(line[1] + "\n")
