with open("./labelThres.txt", "r") as f:
    lines = f.readlines()

#assert len(lines) % 7 == 0

tarlist = [[lines[i + t].strip() for t in range(17)] for i in range(0, len(lines), 17)]
labeled = []

try:
    with open("out.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() != "":
            labeled.append(line.strip())
except Exception as e:
    labeled = []

f = open("out.txt", "w")
for label in labeled:
    f.write(str(label) + "\n")
f.flush()

while len(labeled) < len(tarlist):

    print ("DATA:" + str(len(labeled)) + "/" + str(len(tarlist)))
    tar = tarlist[len(labeled)]
    for k in tar[:14]:
        print (str(k) + "\n")

#    print ("1 for 'good!'")
#    print ("2 for 'bug!'")
#    print ("3 for 'test case error!'")
    a = -1
    b = -1 
    while a != 1 and a != 2 and a != 3:
        try:
            print ("Input 1, 2 or 3:")
            a = int(input())
        except:
            print ("Wrong Input!!! Input again.")
            pass
    
    print (str(tar[14]) + "\n")
    print (str(tar[15]) + "\n")
    b = a
    a = -1
    while a != 1 and a != 2 and a != 3:
        try:
            print ("Input 1, 2 or 3:")
            a = int(input())
        except:
            print ("Wrong Input!!! Input again.")
            pass
    f.write(str(b) + " " + str(a) + "\n")
    f.flush()
#    labeled.append(int(a))
    labeled.append(str(b) + " " + str(a) + "\n")

f.close()


