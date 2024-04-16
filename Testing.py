import os

Genderdirs = ['./NMT_zh_en0-8Mu/Google-Gender', './NMT_zh_en0-8Mu/T5-Gender', './NMT_zh_en0-8Mu/Transformer-Gender']
Countrydirs = ['./NMT_zh_en0-8Mu/Google-Country', './NMT_zh_en0-8Mu/T5-Country', './NMT_zh_en0-8Mu/Transformer-Country']
os.system("python3 ./MutantGen-Test.py")
for gen, cou in zip(Genderdirs, Countrydirs):
    os.system(f"cp NewThres/TestGen-gender/*.txt NewThres/TestGen-gender/*.index {gen}")
    os.system(f"cd {gen} && sh desp.sh")
    os.system(f"cd {gen} && python3 lookupTrans.py")
    os.system(f"cd {gen} && sh test.sh")
    
    os.system(f"cp NewThres/TestGen-country/*.txt NewThres/TestGen-country/*.index {cou}")
    os.system(f"cd {cou} && sh desp.sh")
    os.system(f"cd {cou} && python3 lookupTrans.py")
    os.system(f"cd {cou} && sh test.sh")
