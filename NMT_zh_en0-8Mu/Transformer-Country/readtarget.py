with open("./target.txt") as f:
    lines = f.readlines()

data = [[lines[i].strip(), lines[i + 2].strip()] for i in range(0, len(lines), 7)]

with open("./out.txt") as f:
    lines = f.readlines()

labels = [int(lines[i]) for i in range(len(lines))]
data = data[:len(labels)]
dic = {}
with open("./Com_ALL.txt") as f:
    lines = f.readlines()

for i in range(0, len(lines), 15):
    index = "!@#".join([lines[i + 12].strip(), lines[i + 10].strip()])
    if "nan" not in lines[i + 8]:
        score = eval(lines[i + 8].replace("BERT:", "").replace("False", ""))
    else:
        score = float([])
    dic[index] = score
    print (score)
#exit()
#with open("./")
#for q in range()
q = -1
mf1 = 0
w1 = 0 
#w2 = 0
while w1 <= 1:
    q = 0.95
    while q < 1:
        fp = 0 
        fn = 0
        tp = 0
        tn = 0 
        for i in range(len(data)):
            k = "!@#".join(data[i])
            assert k in dic
            score = w1 * dic[k][0] + (1 - w1) * dic[k][1] - q
            if score >= 0 and labels[i] == 2:
                fn += 1
            if score < 0 and labels[i] == 2:
                tp += 1
            if score >= 0 and labels[i] == 1:
                tn += 1
            if score < 0 and labels[i] == 1:
                fp += 1
        assert fp + fn + tp + tn == 101
        try:
            p = tp / (tp + fp)
        except:
            p = 0
        try:
            r = tp / (tp + fn)
        except:
            r = 0
        try:
            f1 = 2 * p * r / (p + r)
        except:
            f1 = 0
        acc = (tp + tn) / (tp + fp + fn + tn)
        if f1 > mf1:
            mf1 = f1
            bestq = q
            bestw1 = w1
            bestp = p
            bestacc = acc
        print (q, w1, f1, mf1, p, acc)
        q += 0.00001
    w1 += 0.01

print (bestq, bestw1, bestp, mf1, bestacc)
