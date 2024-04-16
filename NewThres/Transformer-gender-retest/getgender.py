import jieba

with open("./finalscore.txt") as f:
    lines = f.readlines()

DATA = [[lines[i + t].strip() for t in range(8)] for i in range(0, len(lines), 8)]

with open("spn.txt") as f:
    lines = f.readlines()

spn = eval(lines[0])

male = {}
female = {}
for data in DATA:
    score = 0.80
    scs = eval(data[-1])
    if scs[0] < score and not any([sc < score for sc in scs[1:]]):
        sen1en = data[2].strip() #.split("\t")[2:]
        sen2en = data[4].strip() #.split("\t")[2:]
        used = []
        sen1zh = jieba.cut(data[1].strip(), cut_all = False)
        sen2zh = jieba.cut(data[3].strip(), cut_all = False)

        if "Gen:\tmale\t" in sen1en:
            for word in sen1zh:
                if word in male:
                    if word not in used:
                        male[word] += 1
                else:
                    male[word] = 1
                used.append(word)
        
        if "Gen:\tfemale\t" in sen1en:
            for word in sen1zh:
                if word in female:
                    if word not in used:
                        female[word] += 1
                else:
                    female[word] = 1
                used.append(word)

        if "Gen:\tmale\t" in sen2en:
            for word in sen2zh:
                if word in male:
                    if word not in used:
                        male[word] += 1
                else:
                    male[word] = 1
                used.append(word)

        if "Gen:\tfemale\t" in sen2en:
            for word in sen2zh:
                if word in female:
                    if word not in used:
                        female[word] += 1
                else:
                    female[word] = 1
                used.append(word)

last = {}

for word in male:
    last[word] = 0
    if word not in female:
        female[word] = 0

for word in female:
    last[word] = 0
    if word not in male:
        male[word] = 0

for word in last:
    last[word] = male[word] - female[word]

l = []
for word in last:
    if word in spn:
        continue
    if abs(last[word]) < 10:
        continue
    l.append([last[word], word])

with open("./female.txt", "w") as f:
    for word in last:
        if last[word] < 0:
            for i in range(-last[word]):
                f.write(f"{word} ")

with open("./male.txt", "w") as f:
    for word in last:
        if last[word] > 0:
            for i in range(last[word]):
                f.write(f"{word} ")


l = sorted(l)
print (l)
