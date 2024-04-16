with open("./f_en_mu.txt", "r") as f:
    lines = f.readlines()

count = 0 

for line in lines:
    if "Gen:\t" in line:
        count += 1

print (count/2)
