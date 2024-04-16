from transformers import AutoTokenizer
from tqdm import tqdm

tknz = AutoTokenizer.from_pretrained('microsoft/CodeGPT-small-py')

#for i in range(1, 18599):
with open(f"./result_theorem.txt", encoding="iso-8859-1") as f:
    lines = f.readlines()
#with open(f"tkCoq.txt", "w", encoding="iso-8859-1") as f:
with open(f"tkCoq.txt", "w") as f:
    for line in tqdm(lines):
        if line.strip() == "":
            continue
        n = " ".join(tknz.tokenize(line))
        f.write(n + "\n")
