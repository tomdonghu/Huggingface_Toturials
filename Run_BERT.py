import torch
import numpy as np
import pandas as pd
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
from tqdm import tqdm
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch import nn
from transformers import AdamW
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm




CVAP_all_SD_df = pd.read_csv('./ChineseEmoBank/CVAP_SD/CVAP_all_SD.csv', encoding= 'utf-8',sep="\t")
df = CVAP_all_SD_df.drop(['No.','Valence_SD', 'Arousal_SD'], axis= 1)


# 提取特徵和標籤
#X = df[['Valence_Mean', 'Arousal_Mean']]
#y = df['Phrase'] # 如果您的數據集中有標籤列，請替換 'label_column_name' 為您的標籤列名稱
x = df['Phrase']
y = df[['Valence_Mean']] # 如果您的數據集中有標籤列，請替換 'label_column_name' 為您的標籤列名稱

# 將數據集分成訓練集和測試集，以 80:20 的比例分割
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)



# bert = AutoModel.from_pretrained('bert-base-chinese', return_dict=False)
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
bert = AutoModel.from_pretrained('ckiplab/bert-base-chinese', return_dict=False)

train_idx = x_train.dropna().index
test_idx = x_test.dropna().index

train_tokens = tokenizer.batch_encode_plus(x_train[train_idx].to_list(),
                                           max_length = 50,
                                           pad_to_max_length = True,
                                           truncation = True)
test_tokens = tokenizer.batch_encode_plus(x_test[test_idx].to_list(),
                                           max_length = 50,
                                           pad_to_max_length = True,
                                           truncation = True)
print(y_train['Valence_Mean'])
# y_train = y_train.reset_index(drop = True)

train_seq = torch.tensor(train_tokens['input_ids'])
train_mask = torch.tensor(train_tokens['attention_mask'])
# print([i for i in y_train['Valence_Mean']])
train_y = torch.tensor([i for i in y_train['Valence_Mean']])

test_seq = torch.tensor(test_tokens['input_ids'])
test_mask = torch.tensor(test_tokens['attention_mask'])
test_y = torch.tensor([i for i in y_test['Valence_Mean']])

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
trainloader = DataLoader(train_data, 
                         sampler = train_sampler,
                         batch_size = 32)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = RandomSampler(test_data)
testloader = DataLoader(test_data, 
                         sampler = test_sampler,
                         batch_size = 32)

for param in bert.parameters():
    param.requires_grad = False

class BertRegressor(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 128)  # add a linear layer with output size 128
        self.relu = nn.ReLU()  # add ReLU activation function
        self.fc2 = nn.Linear(128, 128)  # output one continuous value
        self.fc3 = nn.Linear(128, 1)  # output one continuous value

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)  # apply ReLU activation
        x = self.fc2(x)
        x = self.relu(x)  # apply ReLU activation
        return self.fc3(x).squeeze()  # remove the last dimension of size 1

model = BertRegressor(bert)
model = model.cuda()

optimizer = AdamW(model.parameters(), lr=1e-5)

criterion = nn.MSELoss()

epochs = 20

for e in range(epochs):   
    train_loss = 0.0
    for batch in tqdm(trainloader):
        batch = [i.cuda() for i in batch]
        sent_id, masks, labels = batch

        optimizer.zero_grad()
        preds = model(sent_id, masks)
        loss = criterion(preds, labels)
        train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
    print(f'Epoch:{e+1}\t\tTraining Loss: {train_loss / len(trainloader)}')

pred_label = []
true_label = []
for batch in tqdm(testloader):
    batch = [i.cuda() for i in batch]
    sent_id, masks, labels = batch

    preds = model(sent_id, masks)
    #pred_label.extend(torch.argmax(preds, axis = 1).cpu())
    pred_label.extend(preds.cpu())
    true_label.extend(labels.cpu())

from sklearn.metrics import mean_absolute_error

pred_label = []
true_label = []
for batch in tqdm(testloader):
    batch = [i.cuda() for i in batch]
    sent_id, masks, labels = batch

    preds = model(sent_id, masks)
    pred_label.extend(preds.detach().cpu().numpy())
    true_label.extend(labels.detach().cpu().numpy())

mae = mean_absolute_error(true_label, pred_label)
print(f'MAE: {mae}')

def predict_sentiment(sentence, model, tokenizer):
    encoded_sent = tokenizer.encode_plus(
        sentence,
        truncation=True,
        max_length=50,
        add_special_tokens=True,
        # pad_to_max_length=True,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = encoded_sent['input_ids'].to(device)
    attention_mask = encoded_sent['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    return output.item()

sentence = "我有點不喜歡！"
score = predict_sentiment(sentence, model, tokenizer)
print(score)

