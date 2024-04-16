from tqdm import tqdm

k = []
with open("../LookUpTable.txt") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        k.append(lines[i].strip())

with open("./f_en_mu.txt") as f:
    lines = f.readlines()

used = []
with open("./input.txt", "w") as f:
    lines = [line.strip() for line in lines]
    for i in tqdm(range(len(lines))):
        line = lines[i].replace("Gen:\tfemale\t", "").replace("Gen:\tmale\t", "")
        if line not in used and line not in k:
            used.append(line)
            f.write(line + "\n")

