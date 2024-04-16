with open("outZeyu.txt", "r") as f:
	zylines = f.readlines()

with open("outJie.txt") as f:
	zjlines = f.readlines()

with open("target.txt") as f:
	datalines = f.readlines()

zjlines = [line.strip() for line in zjlines]
zylines = [line.strip() for line in zylines]
data = []
for i in range(0, len(datalines), 7):
	data.append([datalines[i + t].strip() for t in range(6)])

assert len(zylines) == len(zjlines)

for i in range(len(zylines)):
	if zylines[i] != zjlines[i]:
		print (data[i][0])
		print (data[i][1])
		print (data[i][2])
		print (data[i][3])
		print (data[i][4])
		print (data[i][5])
		print ("Jie: ", zjlines[i])
		print ("Zeyu: ", zylines[i])
		print ("\n")

