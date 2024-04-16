dic = {'Norway': 0, 'Belgium': 1, 'Sweden': 2, 'Finland': 3, 'Australia': 4, 'USA': 5, 'Brazil': 6, 'Israel': 7, 'Afghanistan': 8, 'Somalia': 9, 'The Netherlands': 10, 'Ireland': 11, 'Slovenia': 12, 'Greece': 13, 'Latvia': 14, 'Canada': 15, 'Romania': 16, 'Turkey': 17, 'Iran': 18, 'UK': 19, 'Czech': 20, 'India': 21, 'Hungary': 22, 'Ukraine': 23, 'Poland': 24}

l = [0] * 25

with open("./finalCom_BERT.txt") as f:
    lines = f.readlines()

def add(en):
    global l
    cy = en.split('\t')[1].strip().replace('female', '').replace('male', '')
    l[dic[cy]] += 1

for i in range(0, len(lines), 8):
    add(lines[i + 2].strip())
    add(lines[i + 4].strip())

print (l)
print (sum(l))
