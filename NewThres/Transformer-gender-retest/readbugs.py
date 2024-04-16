f = open("./Com_ALL.txt", "r")
lines = f.readlines()
f.close()

flcs = open("./bugs_LCS.txt", "w")
fed = open("./bugs_ED.txt", "w")
ftfidf = open("./bugs_TFIDF.txt", "w")
fbleu = open("./bugs_BLEU.txt", "w")
fbert = open("./bugs_BERT.txt", "w")

# 0.963, 0.963, 0.999, 0.906
for i in range(0, len(lines), 15):
    lcs = float(lines[i + 4].strip().split()[-2])
    ed = float(lines[i + 5].strip().split()[-2])
    tfidf = float(lines[i + 6].strip().split()[-2])
    bleu = float(lines[i + 7].strip().split()[-2])
    bert = float(lines[i + 8].strip().split()[-2])
    
    ori = lines[i + 10].strip()
    mu = lines[i + 12].strip()
   
    print (lcs)
    if lcs < 0.963:
        flcs.write(mu + "\n")
        flcs.write(ori + "\n")
    if ed < 0.963:
        fed.write(mu + "\n")
        fed.write(ori + "\n")
    if tfidf < 0.999:
        ftfidf.write(mu + "\n")
        ftfidf.write(ori + "\n")
    if bleu < 0.906:
        fbleu.write(mu + "\n")
        fbleu.write(ori + "\n")
    if bert < 0:
        fbert.write(mu + "\n")
        fbert.write(ori + "\n")

flcs.close()
fed.close()
fbert.close()
ftfidf.close()
fbleu.close()
