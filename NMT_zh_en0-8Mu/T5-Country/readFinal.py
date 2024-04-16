with open("outZeyu.txt", "r") as f:
	zylines = f.readlines()

with open("outJie.txt") as f:
	zjlines = f.readlines()

with open("target.txt") as f:
	datalines = f.readlines()

with open("Diff.txt") as f:
	difflines = f.readlines()

zjlines = [line.strip() for line in zjlines]
zylines = [line.strip() for line in zylines]
# dflines = [line.strip() for line in difflines]

data = []
for i in range(0, len(datalines), 7):
	data.append([datalines[i + t].strip() for t in range(6)])

dfdata = []
for i in range(0, len(difflines), 10):
	dfdata.append([difflines[i + t].strip() for t in range(10)])

assert len(zylines) == len(zjlines)

count = 0
for i in range(len(zylines)):
	# print (data[i][0])
	# print (data[i][1])
	# print (data[i][2])	
	# print (data[i][3])
	# print (data[i][4])
	# print (data[i][5])	
	if zylines[i] != zjlines[i]:
		print (dfdata[count][-2])
		# print (dfdata[count][-1])
		count += 1
	else:
		print (zjlines[i])
		# print (0)

assert count == len(dfdata)