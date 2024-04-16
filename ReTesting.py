import os

Genderdirs = ['./NMT_zh_en0-8Mu/Google-Gender', './NMT_zh_en0-8Mu/T5-Gender', './NMT_zh_en0-8Mu/Transformer-Gender']
GenderRedirs = ['./NewThres/Google-gender-retest', './NewThres/T5-gender-retest', "./NewThres/Transformer-gender-retest"]
Countrydirs = ['./NMT_zh_en0-8Mu/Google-Country', './NMT_zh_en0-8Mu/T5-Country', './NMT_zh_en0-8Mu/Transformer-Country']
CountryRedirs = ['./NewThres/Google-country-retest', './NewThres/T5-country-retest', "./NewThres/Transformer-country-retest"]
os.system("python3 ./MutantGen-Test.py")
for gen, regen, cou, recou in zip(Genderdirs, GenderRedirs, Countrydirs, CountryRedirs):
    os.system(f"cd {regen} && sh gentest.sh")
    os.system(f"cd {regen} && python3 lookupTrans.py")
    os.system(f"cd {regen} && sh test.sh")
    
    os.system(f"cd {recou} && sh gentest.sh")
    os.system(f"cd {recou} && python3 lookupTrans.py")
    os.system(f"cd {recou} && sh test.sh")
