import wordcloud

stp = []
with open("./spn.txt") as f:
    line = f.readline()
    stp = eval(line)
stp.append("and")
stp.append("（")
stp.append("）")
stp.append("，")
stp.append("。")


with open("./female.txt") as f:
    lines = f.readlines()

w = wordcloud.WordCloud(font_path="simfang.ttf", stopwords=stp, width=1000, height=618)
w.generate(" and ".join(lines[0].split()))
w.to_file("female.png")

with open("./male.txt") as f:
    lines = f.readlines()

w = wordcloud.WordCloud(font_path="simfang.ttf", stopwords=stp, width=1000, height=618)
w.generate(" and ".join(lines[0].split()))
w.to_file("male.png")
