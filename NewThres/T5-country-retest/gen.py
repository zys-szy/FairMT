from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from tqdm import tqdm

tokenizer = MT5Tokenizer.from_pretrained("./model/model")
model = MT5ForConditionalGeneration.from_pretrained("./model/model").cuda()

with open("./en_mu.txt", "r") as f:
    lines = f.readlines()

batch_size = 10

with open("./f_en_mu.zh.beam.txt", "w") as f:
    for i in tqdm(range(0, len(lines), batch_size)):
        ls = ["translate English to Chinese: " + lines[t + i].strip() for t in range(min(batch_size, len(lines) - i))]
        tks = tokenizer(ls, return_tensors="pt",padding=True)
        outputs = model.generate(    input_ids=tks["input_ids"].cuda(),
                attention_mask=tks["attention_mask"].cuda(),
                 max_length=300, num_beams=16)
#        print (outputs.size())
        s = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
#        with open("./gen.txt", "w") as f:
#        print (s)
        for k in s:
            f.write(k.strip() + "\n")
        f.flush()
