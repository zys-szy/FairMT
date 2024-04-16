with open("./en_mu.index", "r") as f:
    lines = f.readlines()

s = []
for i in range(len(lines)):
    if lines[i].strip() not in s:
        s.append(lines[i].strip())

print (len(s))
