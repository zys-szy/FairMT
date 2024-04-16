with open("./test.csv") as f:
    lines = f.readlines()

with open("./train.csv") as f:
    lines += f.readlines()

with open("./en_tk.txt", "w") as f:
    for line in lines:
        line = line.split("\t")[1]
        if len(line.split()) <= 200:
            f.write(line.strip() + "\n")

