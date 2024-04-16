from tqdm import tqdm

with open("./f_en_mu.txt") as f:
    lines = f.readlines()

used = []
with open("./input.txt", "w") as f:
    lines = [line.strip() for line in lines]
    for i in tqdm(range(len(lines))):
        line = lines[i].replace("\t".join(lines[i].split("\t")[:2]), "").strip()
        if line not in used:
            used.append(line)
            f.write(line + "\n")

