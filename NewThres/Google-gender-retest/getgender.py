import jieba

with open("./finalCom_BERT.txt") as f:
    lines = f.readlines()

DATA = [[lines[i + t].strip() for t in range(8)] for i in range(0, len(lines), 8)]


femalewords = ['her', 'she', 'daughter']
malewords = ['his', 'him', 'he', 'gentleman']

with open("../asset/gender_associated_word/masculine-feminine.txt") as f:
    lines = f.readlines()

for line in lines:
    if line == "":
        continue
    femalewords.append(line.strip().split(',')[1].strip())
    malewords.append(line.strip().split(',')[0].strip())

with open("../asset/gender_computer/female_names_only_USA.csv") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

femalewords += lines

#print (femalewords)

with open("../asset/gender_computer/male_names_only_USA.csv") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

malewords += lines

with open("spn.txt") as f:
    lines = f.readlines()

spn = eval(lines[0])

male = {}
female = {}

with open("../TestGenerator-NMTMLT-2-large-true/f_en_mu.txt") as f:
    lines = f.readlines()
    kys = [line.strip() for line in lines]
#print (kys)

def readgender(en1, en2):
    ret1 = ""
    ret2 = ""
    for k in kys:
        if en1.strip() in k:
            ret1 = k.split('\t')[1].strip()
        if en2.strip() in k:
            ret2 = k.split('\t')[1].strip()
    #print (en1)
    #print (en2)
    #print (ret1, ret2)
    assert ret1 != '' and ret2 != ''
    return ret1, ret2
    en1tks = en1.split()
    en2tks = en2.split()
    for m, n in zip(en1tks, en2tks):
        m = m.lower()
        n = n.lower()
        if m != n:
            if m in malewords and n in femalewords:
                return "male", "female"
            if n in malewords and m in femalewords:
                return "female", "male"
            print (en1)
            print (en2)
            print (m,n)
            assert False

for data in DATA:
    score = 0.80
    scs = eval(data[-1])
    if scs[0] < score and not any([sc < score for sc in scs[1:]]):
        sen1en = data[2].strip() #.split("\t")[2:]
        sen2en = data[4].strip() #.split("\t")[2:]
        used = []
        sen1zh = jieba.cut(data[1].strip(), cut_all = False)
        sen2zh = jieba.cut(data[3].strip(), cut_all = False)
        gender1, gender2 = readgender(sen1en, sen2en)
        if gender1 == "male":
            for word in sen1zh:
                if word in male:
                    if word not in used:
                        male[word] += 1
                else:
                    male[word] = 1
                used.append(word)
        
        if gender1 == "female":
            for word in sen1zh:
                if word in female:
                    if word not in used:
                        female[word] += 1
                else:
                    female[word] = 1
                used.append(word)

        if gender2 == "male":
            for word in sen2zh:
                if word in male:
                    if word not in used:
                        male[word] += 1
                else:
                    male[word] = 1
                used.append(word)

        if gender2 == "female":
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
    if abs(last[word]) < 5:
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
