with open("./en_mu.txt") as f:
    lines = f.readlines()

with open("./en_mu.ori.txt", "w") as f:
#    lines = f.readlines()
    for line in lines:
        f.write(line.replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", ""))

