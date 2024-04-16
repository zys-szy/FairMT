import string

f = open("en_mu.txt")
lines = f.readlines()
f.close()

comma = string.punctuation

f = open("f_en_mu.txt", "w")
for i in range(0, len(lines), 2):
    ori = lines[i].strip()
    mut = lines[i + 1].strip().split("\t")[0]
    #pro = float(lines[i + 1].strip().split("\t")[1].split()[-1])
    #if pro < 0.01:
    #    continue
    if "[UNK]" in mut:
        continue
    if ori.strip() == mut.strip():
        continue
    if ori == mut:
        continue
    oriTokens = ori.strip().split()
    mutTokens = mut.strip().split()
    
    print ("-------------------")
    print (oriTokens)
    print (mutTokens)
    if len(oriTokens) != len(mutTokens):
        continue
    
    good = True
    for t in range(len(oriTokens)):
        mutToken = mutTokens[t]
        oriToken = oriTokens[t]
        if mutToken != oriToken and (mutToken in comma or oriToken in comma):
            good = False
            break

    if good:
        f.write(ori + "\n")#.replace(" ##", "") + "\n")
        f.write(mut + "\n")#.replace(" ##", "") + "\n")

f.close()
            
