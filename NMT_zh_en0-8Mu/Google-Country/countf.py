with open("./Com_BERT.txt") as f:
    lines = f.readlines()

data = [[lines[i + t].strip() for t in range(7)] for i in range(0, len(lines), 7)]

count85 = 0
count80 = 0
count75 = 0
count70 = 0

for i in range(len(data)):
    score = float(data[i][0][1:].split(",")[0])
    if score <= 0.85:
        count85 += 1
    if score <= 0.80:
        count80 += 1
    if score <= 0.75:
        count75 += 1
    if score <= 0.70:
        count70 += 1

print (count85)
print (count80)
print (count75)
print (count70)
