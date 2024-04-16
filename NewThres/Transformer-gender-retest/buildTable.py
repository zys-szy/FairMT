with open("./en_mu.txt") as f: #./NewThres/TestGenerator-NMTRep/en_mu.txt") as f:
    lines = f.readlines()

with open("./f_en_mu.zh.beamtttt") as f:
    blines = f.readlines()

with open("./LookUpTable.txt", "w") as f:
    for i in range(len(blines)):
        f.write(lines[i])
        f.write(blines[i].replace(" ", ""))
