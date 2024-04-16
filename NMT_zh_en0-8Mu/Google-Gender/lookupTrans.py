f = open("./en_mu.txt")
enlines = f.readlines()
f.close()

enlines = [enline.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "").strip() for enline in enlines]

f = open("./LookUpTable.txt")
lines = f.readlines()
f.close()

dic = {}
for t in range(0, len(lines), 2):
    dic[lines[t].strip()] = lines[t + 1].strip()

f = open("./f_en_mu.zh.beam", "w")
ff = open("./candidata.en", "w")
used = []
for line in enlines:
    if line not in dic:
        if line in used:
            continue#ff.write(line.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "") + "\n")
        ff.write(line.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "") + "\n")
        used.append(line)
        continue
    f.write(dic[line.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "")] + "\n")
f.close()
ff.close()
