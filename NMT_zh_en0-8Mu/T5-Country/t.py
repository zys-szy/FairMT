import re

a = "['这些 3500 个 字 的 文件 中 ， 这些 司法 和 自由 的 战斗人员 是 对 约翰 的 任意 扣押 其 臣民 财产 和 人 的 简短 陈述 。   {+Magna   Carta+} 在 第 39 章中 [-， Magna   Carta-] 指出 ： 任何 自由 [-兄弟-] {+女孩+} 都 不会 [-被 逮捕-] {+被捕+} ， [-被-] 监禁 [-， 被 监禁 ， 被 剥夺 了 [-] {+或 肆虐 （+} 被 剥夺 [-]-] {+）+} ， 被 宣告 或 被 宣告 或 流放 ， [-或-] {+或者+} 以 任何 方式 被 摧毁 ， 我们 也不会 反对 [-他-] {+她+} ， 我们 也 不会 派遣 。   [-他-]   {+她+} ， 根据 [-他-] {+她+} 的 同龄人 或 土地 法律 的 合法 判决 。']"

def sentences_from_wdiff(wdiff_line):
    count_o = []
    count_n = []
    bf = ""
    #for line in lines:
    now = wdiff_line
    now = bf + " " + now
    now = re.sub(r"([\[])([\-])", r" \1\2 ", now)
    now = re.sub(r"([\-])([\]])", r" \1\2 ", now)
    now = re.sub(r"([\{])([\+])", r" \1\2 ", now)
    now = re.sub(r"([\+])([\}])", r" \1\2 ", now)
    now = now.replace("[- ]", "[-]")
    print (now)

    # stable
    o = []
    n = []
    #print ("-----------------")
    words = now.strip().split()
    old_tokens = []
    new_tokens = []

    in_old = 0
    in_new = 0
    tokens = words
    for i in range(len(words)):
        if in_old > 0 and tokens[i] not in ["{+", "+}", "[-", "-]"]:
            old_tokens.append(tokens[i])
        elif in_new > 0 and tokens[i] not in ["{+", "+}", "[-", "-]"]:
            new_tokens.append(tokens[i])
        elif tokens[i] not in ["{+", "+}", "[-", "-]"]:
            old_tokens.append(tokens[i])
            new_tokens.append(tokens[i])

        if tokens[i] == "{+":
            in_new += 1
        elif tokens[i] == "+}":
            in_new -= 1

        if tokens[i] == "[-":
            in_old += 1
        elif tokens[i] == "-]":
            in_old -= 1

        assert in_new < 2 and in_new > -1 and in_old < 2 and in_old > -1 and in_new + in_old < 2
    o.append(old_tokens)
    n.append(new_tokens)
    #o.append(words)
    #n.append(words)
    #print (" ".join(words))
    for i in range(len(words)):
        if words[i] == "[-":
            for t in range(i + 1, len(words)):
                if words[t] == "-]":
                    sentence = []
                    in_new = 0
                    if t - i >= 5:
                        break
                    for k in range(len(words)):
                        if k >= i and k <=t:
                            continue
                        elif words[k] == "{+":
                            in_new += 1
                        elif words[k] == "+}":
                            in_new -= 1
                            assert in_new >= 0
                            continue
                        if in_new > 0:
                            continue
                        if words[k] in ["[-", "-]"]:
                            continue
                        else:
                            sentence.append(words[k])
                    #print (" ".join(sentence))
                    o.append(sentence)
                    break

        if words[i] == "{+":
            for t in range(i + 1, len(words)):
                if words[t] == "+}":
                    sentence = []
                    in_new = 0
                    if t - i >= 5:
                        break
                    for k in range(len(words)):
                        if k >= i and k <=t:
                            continue
                        elif words[k] == "[-":
                            in_new += 1
                        elif words[k] == "-]":
                            in_new -= 1
                            assert in_new >= 0
                            continue
                        if in_new > 0:
                            continue
                        if words[k] in ["{+", "+}"]:
                            continue
                        else:
                            sentence.append(words[k])
                    #print (" ".join(sentence))
                    n.append(sentence)
                    break

    for i in o:
        for t in n:
            count_o.append(i)
            count_n.append(t)

    return count_o, count_n 

sentences_from_wdiff(a)
