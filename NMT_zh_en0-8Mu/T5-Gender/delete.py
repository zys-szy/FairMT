with open("./target.txt", "r") as f:
    lines = f.readlines()

assert len(lines) % 7 == 0

tarlist = [[lines[i + t]for t in range(7)] for i in range(0, len(lines), 7)]

with open("./newtarget.txt", "w") as f:
    for i in range(len(tarlist)):
        if (i >= 100 and i < 200) or (i >= 300 and i < 400) or (i >= 500 and i < 600):
            continue
        for k in tarlist[i]:
            f.write(k)
            
