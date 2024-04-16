with open("../../NewThres/asset/gender_computer/female_names_only_USA.csv") as f:
    females = [line.strip() for line in f.readlines()][1:]

with open("../../NewThres/asset/gender_computer/male_names_only_USA.csv") as f:
    males = [line.strip() for line in f.readlines()][1:]

with open("../../NewThres/asset/gender_computer/unique_female_names_and_country.csv") as f:
    femalescty = [line.strip().split(",")[1] for line in f.readlines()][1:]

with open("../../NewThres/asset/gender_computer/unique_male_names_and_country.csv") as f:
    malescty = [line.strip().split(",")[1] for line in f.readlines()][1:]

with open("./Com_BERT.txt") as f:
    lines = f.readlines()

data = [[lines[i + t].strip() for t in range(7)] for i in range(0, len(lines), 7)]

count85 = 0
count80 = 0
count75 = 0
count70 = 0

for i in range(len(data)):
    sent1 = data[i][2].strip()
    sent2 = data[i][4].strip()
    fename = 0 
    maname = 0 
    for name in femalescty:
        if name in sent1 and name not in sent2:
            fename += 1
        elif name not in sent1 and name in sent2:
            fename += 1
    for name in malescty:
        if name in sent1 and name not in sent2:
            maname += 1
        elif name not in sent1 and name in sent2:
            maname += 1
    if (fename >= 2 or maname >= 2):
        continue

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
