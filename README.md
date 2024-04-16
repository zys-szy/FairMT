# FairMT (Fairness Testing of Machine Translation Systems)

Machine translation is integral to international communication and extensively employed in diverse human-related applications. Despite remarkable progress, fairness issues persist within current machine translation systems. In this paper, we propose FairMT, an automated fairness testing approach tailored for machine translation systems. FairMT operates on the assumption that translations of semantically similar sentences, containing protected attributes from distinct demographic groups, should maintain comparable meanings. It comprises three key steps: (1) test input generation, producing inputs covering various demographic groups; (2) test oracle generation, identifying potential unfair translations based on semantic similarity measurements; and (3) regression, discerning genuine fairness issues from those caused by low-quality translation. Leveraging FairMT we conduct an empirical study on three leading machine translation systems—Google Translate, T5, and Transformer.

For Testing:
```
python3 Testing.py
```
After it, the results are in the ```NMT_zh_en0-8Mu/``` folder.

For ReTesting:
```
python3 ReTesting.py
```
After it, the results are in the ```NewThres/xxx-xxx-retest``` folder (```finalCom_BERT.txt```). 

# Dependenices
```
NLTK 3.2.1
Pytorch 1.6.1
Python 3.7
Ubuntu 16.04
Transformers 3.3.0
```
