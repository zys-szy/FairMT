with open("final.txt") as f:
	lines = f.readlines()

def labelSet (index, line, TF):
	if int(line) in [1, 3] and index % 2 ==0:
		TF["FP"] += 1
	elif int(line) in [2] and index % 2 ==0:
		TF["TP"] += 1
	elif int(line) == 1 and index % 2 ==1:
		TF["TN"] += 1
	elif int(line) in [2, 3] and index % 2 ==1:
		TF["FN"] += 1

def comSet (TF):
	s = 0
	for k in TF:
		s += TF[k]

	for k in TF:
		TF[k] /= s
	return TF
	

Trans = {"FP":0, "TP":0, "TN":0, "FN":0}
New = {"FP":0, "TP":0, "TN":0, "FN":0}
SIT = {"FP":0, "TP":0, "TN":0, "FN":0}

for i in range(0, 100):
	labelSet(i, lines[i], SIT)

for i in range(100, 200):
	labelSet(i, lines[i], New)

for i in range(200, 300):
	labelSet(i, lines[i], Trans)

Trans = comSet(Trans)
New = comSet(New)
SIT = comSet(SIT)

print ("SIT", SIT)
print ("New", New)
print ("TransRepair", Trans)
