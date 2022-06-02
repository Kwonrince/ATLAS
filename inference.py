import datasets
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import dataset
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

rouge = datasets.load_metric("rouge")

train_data = dataset.BartDataset()
validation_data = dataset.BartDataset(val=True)
test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split='test[:500]')

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').cuda() # ainize/bart-base-cnn
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

ckpt_ = torch.load('./save_base/12_3.4598.pt', map_location=torch.device('cuda'))
ckpt = torch.load('./save/12_3.4372.pt', map_location=torch.device('cuda'))

model.load_state_dict(ckpt_['state_dict'])
model.load_state_dict(ckpt['state_dict'])

#%%
idx = 0
d = validation_data[idx][0].unsqueeze(0)
summary_ids = model.generate(d.cuda(), min_length=56, max_length=142, length_penalty=2.0, early_stopping=True, num_beams=4, no_repeat_ngram_size=3)
summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
summary

t = validation_data[idx][4]
target = tokenizer.decode(t.tolist()[:np.where(t==-100)[0][0]], skip_special_tokens=True)
target

scorer.score(summary, target)

rouge.compute(predictions = [summary], references = [target], rouge_types=['rouge1','rouge2','rougeL'])
#%% huggingface rouge
idx = 0
rouge1, rouge2, rougeL = [], [], []
for idx in tqdm(range(500)):
    d = tokenizer.encode(test_data['article'][idx])
    if len(d) > 1024:
        d = d[:1023] + [2]
    summary_ids = model.generate(torch.tensor([d]).cuda(), min_length=56, max_length=142, length_penalty=2.0, early_stopping=True, num_beams=4, no_repeat_ngram_size=3)
    summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    target = test_data['highlights'][idx]

    summary
    target
    
    rouge.add_batch(predictions=[summary], references=[target])
    
    scores = scorer.score(summary, target)
    rouge1.append(scores['rouge1'][2])
    rouge2.append(scores['rouge2'][2])
    rougeL.append(scores['rougeL'][2])

score = rouge.compute(rouge_types=['rouge1','rouge2','rougeL'])

epoch_500 = score

r1_500 = np.array(rouge1).mean()
r2_500 = np.array(rouge2).mean()
rl_500 = np.array(rougeL).mean()


#%% rouge_score
rouge1, rouge2, rougeL = [], [], []
for idx in tqdm(range(len(validation_data))):
    d = validation_data[idx][0].unsqueeze(0)
    summary_ids = model.generate(d.cuda(), min_length=56, max_length=142, length_penalty=2.0, early_stopping=True, num_beams=4, no_repeat_ngram_size=3)
    summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

    t = validation_data[idx][4]
    target = tokenizer.decode(t.tolist()[:np.where(t==-100)[0][0]], skip_special_tokens=True)

    scores = scorer.score(summary, target)
    rouge1.append(scores['rouge1'][2])
    rouge2.append(scores['rouge2'][2])
    rougeL.append(scores['rougeL'][2])

r1 = np.array(rouge1).mean()
r2 = np.array(rouge2).mean()
rl = np.array(rougeL).mean()

