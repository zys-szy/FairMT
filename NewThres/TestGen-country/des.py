with open("./f_en_mu.txt", "r") as f:
    lines = f.readlines()

with open("./en_mu.txt", "w") as f:
    for line in lines:
        f.write(line.replace(" ##", ""))

