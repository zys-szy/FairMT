f = open("./Com_ALL.txt", "r")
lines = f.readlines()
f.close()

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
   
    if bert < 0:
        fbert.write(mu + "\n")
        fbert.write(ori + "\n")

fbert.close()
