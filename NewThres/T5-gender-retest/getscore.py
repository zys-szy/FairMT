with open("./Com_BERT.txt", "r") as f:
    lines = f.readlines()

DATA = [[lines[i + 4].strip().replace(" ", "").replace("\t", "")] + [(lines[i + 2].strip() + lines[i + 4].strip()).replace(" ", "").replace("\t", "")] + [lines[i + t].strip() for t in range(7)] for i in range(0, len(lines), 7)]


with open("./Com_BERT_F.txt", "r") as f:
    lines = f.readlines()

DATAori = [[lines[i + 4].strip().replace(" ", "").replace("\t", "").replace("Gen:male", "").replace("Gen:female", "")] + [(lines[i + 2].strip() + lines[i + 4].strip()).replace(" ", "").replace("\t", "").replace("Gen:male", "").replace("Gen:female", "")] + [lines[i + t].strip() for t in range(7)] for i in range(0, len(lines), 7)]

def getscore(data):
    return float(data[2].split()[0].replace("[", "").replace(",", ""))

dictori = {}
print (len(DATAori))
print (len(DATA))
for data in DATAori:
    dictori[data[0].replace("Gen:male", "").replace("Gen:female", "")] = [data, [getscore(data)]]
#    print (data[0].replace("Gen:male", "").replace("Gen:female", ""))
#    print (data[0])

for data in DATA:
    if data[0] in dictori:
        print (2)
        if data[1] != dictori[data[0]][0][1]:
            #print (1)
            dictori[data[0]][1].append(getscore(data))

with open("./finalscore.txt", "w") as f:
    for item in dictori:
        for k in dictori[item][0][2:]:
            f.write(k + "\n")
        
        f.write(str(dictori[item][1]) + "\n")
#print ()
