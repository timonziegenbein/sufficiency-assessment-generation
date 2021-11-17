from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats

df_train = pd.read_json('../../../ceph_data/intermediate/corpus-ukp-argument-annotated-essays-v2/aae3.json')
df_split = pd.read_csv('../../../ceph_data/input/UKP-InsufficientArguments_v1.0/data-splitting.tsv', sep='\t', names=['index']+[str(i) for i in range(100)], index_col=False)

def make_sentence(x):
    x = x.capitalize()
    if x[-1] not in ['.','?','!']:
        x += '.'
    return x

df_train['premises'] = df_train['premises'].apply(lambda x: ' '.join([make_sentence(x1) for x1 in x]))
df_train['conclusion'] = df_train['conclusion'].apply(lambda x: make_sentence(x))
df_train['both'] = df_train[['conclusion', 'premises']].apply(lambda x: ' '.join(x), axis=1)
proj_dict = {0:-1,1:1}

predictions = []
for i in range(10):
    split_dict = dict(zip(df_split['index'], df_split[str(i)]))
    df_train['split'] = df_train['index'].apply(lambda x: split_dict[x])
    tmp_predictions = []
    sentences = df_train['premises'].values
    batch_size = 16
    tokenizer = BertTokenizer.from_pretrained('../../../ceph_data/output/bert-AAE-v2-only-dot-direct-cola-only-premises/{}/best_tfmr2'.format(i))
    model = BertForSequenceClassification.from_pretrained('../../../ceph_data/output/bert-AAE-v2-only-dot-direct-cola-only-premises/{}/best_tfmr2'.format(i), return_dict=True).cuda()
    pbar = tqdm(range(batch_size,len(sentences)+batch_size,batch_size))

    for i in pbar:
        pbar.set_description('|   -running model')
        if i<len(sentences):
            sentences_batch = sentences[i-batch_size:i]
        else:
            sentences_batch = sentences[i-batch_size:]

        input_tokens = tokenizer(list(sentences_batch), max_length=512, return_tensors='pt', truncation=True, padding=True, add_special_tokens=True)
        outputs = model(input_tokens['input_ids'].cuda())
        tmp_predictions += list(torch.argmax(torch.nn.functional.softmax(outputs.logits), dim=1).cpu().numpy())
    df_train['preds'] = tmp_predictions
    df_train['preds'] = df_train[['split', 'preds']].apply(lambda x: proj_dict[x[1]] if x[0]=='TEST' else 0, axis=1)
    predictions.append(list(df_train['preds'].values))
predictions = np.sum(predictions, axis=0)
#print(predictions)

df_train['scores'] = predictions
df_train[['premises', 'conclusion','scores','local_sufficency']].to_csv('../../../ceph_data/output/bert-AAE-v2-only-dot-direct-cola-only-premises/aae_preds_test.csv')

        # inputs = tokenizer("""Sustaining the cultural values of immigrants is paramount essential. Maintaining one’s cultural identity is a key important rule to help individuals emerge in the new multicultural environments. Take australia for example, immigrants from varieties of nations have a day called multicultural day where people from each country prepare their food and traditional activities for displaying in the public venues. Many australians come this day to enjoy the shows, learn about the cultures and admire the diverse values. These feedbacks, in turn, help raise one’s pride of their cultures and help people understand each other more.""", return_tensors="pt")
        # outputs = model(**inputs)
        # loss = outputs.loss
        # logits = outputs.logits

        # print(torch.argmax(torch.nn.functional.softmax(outputs.logits), dim=1))


# '/workspace/ceph_data/output/bert-icle-dor-direct-cola-labels'
# '/workspace/ceph_data/input/icle_ds/train.json'
# '/workspace/ceph_data/input/icle_ds/valid.json'