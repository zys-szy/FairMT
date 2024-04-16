import sys

used = []
fff = open("./finalCom_BERT.txt", "w")
for i in range(70, 90, 5):
    tarscore = float(i) * 0.01
    with open("./finalscore.txt") as f:
        lines = f.readlines()

    data = [[lines[i + t] for t in range(8)] for i in range(0, len(lines), 8)]

#    f = open("./finalCom_BERT.txt", "w")

    count = 0
    testcases = 0
    for da in data:
        score = eval(da[7])
        sent1 = da[2].strip()
        sent2 = da[4].strip()
        flag = False
        tc = 0
        if score[0] >= tarscore:
            flag = True
            continue
        for i in score[1:]:
            if float(i) < tarscore:
                flag = True
            testcases += 1
        if flag:
            continue
        fename = 0
        maname = 0
        if da not in used:
            for d in da:
                fff.write(d.strip() + "\n")
            count += 1
            used.append(da)
#        testcases += tc

    print (count)
    print (testcases)

    #for line in lines:
fff.close()
