with open("./DATA.txt") as f:
    lines = f.readlines()

data = []
for i in range(0, len(lines), 12):
    data.append([lines[i + t].strip() for t in range(12)])

scores = [[0, 0, []]]
scores += [[i, 0, []] for i in range(50, 100, 5)]
for k in data:
    s = int(float(k[1]) * 100)
    for sc in scores[::-1]:
        if s > sc[0]:
            sc[1] += 1
            sc[2].append(k)
            break

for i in range(len(scores)):
    print (scores[i][0], scores[i][1])

import random

with open("./labelThres.txt", "w") as f:
    for i in range(len(scores)):
        random.shuffle(scores[i][2])
        for k in scores[i][2][:10]:
            for line in k:
                f.write(line + "\n")
            f.write("Semantic similar? (1 for different; 2 for similar; 3 for unknown case (some cases that may contain errors))\n")
            f.write("Label: \n")
            f.write("Fairness? (1 for have fairness problem; 2 for good; 3 for unknown case)\n")
            f.write("Label: \n\n")


