f = open("./f_en_mu.txt")
orienlines = f.readlines()
f.close()
f = open("./f_en_mu.txt")
enlines = f.readlines()
f.close()

#orienlines = []
enlines = [enline.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "").strip() for enline in enlines]

f = open("./LookUpTable.txt")
lines = f.readlines()
f.close()

dic = {}
for t in range(0, len(lines), 2):
    dic[lines[t].strip()] = lines[t + 1].strip()


print (len(dic))
#print (dic)
fin = open("./en_mu.txt", "w")
finori = open("./f_en_mu.ori.txt", "w")
f = open("./f_en_mu.zh.beam", "w")
ff = open("./candidata.en", "w")
used = []
for i in range(0, len(enlines), 2):
    if enlines[i] not in dic or enlines[i + 1] not in dic:
        continue
    
    #if line not in dic:
    #    if line in used:
    #        continue#ff.write(line.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "") + "\n")
    #    ff.write(line.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "") + "\n")
    #    used.append(line)
    #    continue
    finori.write(orienlines[i])
    finori.write(orienlines[i + 1])
    fin.write(enlines[i] + "\n")
    fin.write(enlines[i + 1] + "\n")
    f.write(dic[enlines[i].replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "")] + "\n")
    f.write(dic[enlines[i + 1].replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "")] + "\n")
f.close()
ff.close()
