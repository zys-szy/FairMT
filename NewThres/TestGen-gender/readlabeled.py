with open("./labeled.txt") as f :
    lines = f.readlines()

data = eval(lines[0])

for d in data:
    if d[1] == '0':
        print (d)

