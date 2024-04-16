import jieba

with open("./Com_BERT.txt") as f:
    lines = f.readlines()

lines = [line for line in lines]

male = {}
female = {}
for i in range(0, len(lines), 7):
    score = float(lines[i].split()[0].replace("[", "").replace(",", ""))
    if score >= 0.75:
        continue
    sen1en = lines[i + 2].strip()
    sen2en = lines[i + 4].strip()
    used = []
    sen1zh = jieba.cut(lines[i + 1].strip(), cut_all = False)
    sen2zh = jieba.cut(lines[i + 3].strip(), cut_all = False)

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
    if last[word] == 0:
        continue
    l.append([last[word], word])

print (sorted(l))

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
