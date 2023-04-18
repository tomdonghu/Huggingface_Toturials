import torch
import numpy as np
import pandas as pd
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
from tqdm import tqdm
torch.cuda.is_available()

CVAP_all_SD_df = pd.read_csv('./ChineseEmoBank/CVAP_SD/CVAP_all_SD.csv', encoding= 'utf-8',sep="\t")
df0 = CVAP_all_SD_df.drop(['No.','Valence_SD', 'Arousal_SD'], axis= 1)
df0['class'] = 'Phrase'
#df0.columns = ['Phrase', 'Valence_Mean', 'Arousal_Mean', 'class']

CVAS_all_SD_df = pd.read_csv('./ChineseEmoBank/CVAS_SD/CVAS_all.csv', encoding= 'utf-8',sep="\t")
df1 = CVAS_all_SD_df.drop(['Valence_SD', 'Arousal_SD'], axis= 1)
df1['class'] = 'Text'
df1.columns = ['Phrase', 'Valence_Mean', 'Arousal_Mean', 'class']


CVAW_all_SD_df = pd.read_csv('./ChineseEmoBank/CVAW_SD/CVAW_all_SD.csv', encoding= 'utf-8',sep="\t")
df2 = CVAW_all_SD_df.drop(['No.','Valence_SD', 'Arousal_SD'], axis= 1)
df2['class'] = 'Word'
df2.columns = ['Phrase', 'Valence_Mean', 'Arousal_Mean', 'class']


#df = CVAP_all_SD_df.drop(['No.','Valence_SD', 'Arousal_SD'], axis= 1)
print(df0.loc[1],df1.loc[1],df2.loc[1])

df = pd.concat([df0,df1,df2], axis= 0).reset_index(drop= True)
#df = df0
df.loc[0]


# 提取特徵和標籤
#X = df[['Valence_Mean', 'Arousal_Mean']]
#y = df['Phrase'] # 如果您的數據集中有標籤列，請替換 'label_column_name' 為您的標籤列名稱
x = df['Phrase']
y = df[['Valence_Mean','Arousal_Mean']] # 如果您的數據集中有標籤列，請替換 'label_column_name' 為您的標籤列名稱

# 將數據集分成訓練集和測試集，以 80:20 的比例分割
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(x_train)


# 提取特徵和標籤
#X = df[['Valence_Mean', 'Arousal_Mean']]
#y = df['Phrase'] # 如果您的數據集中有標籤列，請替換 'label_column_name' 為您的標籤列名稱
x = df['Phrase']
y = df[['Valence_Mean','Arousal_Mean']] # 如果您的數據集中有標籤列，請替換 'label_column_name' 為您的標籤列名稱

# 將數據集分成訓練集和測試集，以 80:20 的比例分割
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(x_train)

train_idx = x_train.dropna().index
test_idx = x_test.dropna().index

train_tokens = tokenizer.batch_encode_plus(x_train[train_idx].to_list(),
                                           max_length = 50,
                                           #pad_to_max_length = True,
                                           padding=True,
                                           truncation = True)
test_tokens = tokenizer.batch_encode_plus(x_test[test_idx].to_list(),
                                           max_length = 50,
                                          # pad_to_max_length = True,
                                           padding=True,
                                           truncation = True)
#print(y_train['Valence_Mean'])
# y_train = y_train.reset_index(drop = True)
#y_train.loc[0]

train_seq = torch.tensor(train_tokens['input_ids'])
train_mask = torch.tensor(train_tokens['attention_mask'])
# print([i for i in y_train['Valence_Mean']])
# train_y = torch.tensor([i for i in y_train['Valence_Mean']])
train_y = torch.tensor([(i, j) for i, j in zip(y_train['Valence_Mean'], y_train['Arousal_Mean'])])
test_seq = torch.tensor(test_tokens['input_ids'])
test_mask = torch.tensor(test_tokens['attention_mask'])
#test_y = torch.tensor([i for i in y_test['Valence_Mean']])
test_y = torch.tensor([(i, j) for i, j in zip(y_test['Valence_Mean'], y_test['Arousal_Mean'])])

from torch.utils.data import TensorDataset, RandomSampler, DataLoader

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

from torch import nn
from transformers import AdamW
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

# class BertRegressor(nn.Module):
#     def __init__(self, bert):
#         super().__init__()
#         self.bert = bert
#         self.fc1 = nn.Linear(768, 1)  # output one continuous value
    
#     def forward(self, sent_id, mask):
#         _, cls_hs = self.bert(sent_id, attention_mask=mask)
#         return self.fc1(cls_hs).squeeze()  # remove the last dimension of size 1
    
class BertRegressor(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 128)  # add a linear layer with output size 128
        self.relu = nn.ReLU()  # add ReLU activation function
        self.fc2 = nn.Linear(128, 128)  # output one continuous value
        self.fc3 = nn.Linear(128, 2)  # output one continuous value

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)  # apply ReLU activation
        x = self.fc2(x)
        x = self.relu(x)  # apply ReLU activation
        # return self.fc3(x).squeeze()  # remove the last dimension of size 1
        return self.fc3(x)

model = BertRegressor(bert)
model = model.cuda()

optimizer = AdamW(model.parameters(), lr=1e-5)

#criterion = nn.MSELoss()
criterion = nn.MSELoss(reduction='sum')

from tqdm import tqdm

epochs = 500
losses = []
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
    losses.append(train_loss / len(trainloader))
    print(f'Epoch:{e+1}\t\tTraining Loss: {train_loss / len(trainloader)}')

from matplotlib import pyplot as plt
plt.plot(losses, label='train_loss')

plt.legend()
plt.show

#在上面的代码中，我们首先定义了两个空列表pred_label和true_label来存储
#模型的预测标签和真实标签。然后，我们遍历测试集并对每个批次进行预测
#。将预测值和真实值添加到相应的列表中后，我们可以使用
#sklearn库中的mean_absolute_error函数来计算MAE。
#最后，我们将MAE打印出来。
from sklearn.metrics import mean_absolute_error

# pred_label = []
# true_label = []
pred_label_1 = []
pred_label_2 = []
true_label_1 = []
true_label_2 = []
for batch in tqdm(testloader):
    batch = [i.cuda() for i in batch]
    sent_id, masks, labels = batch

    preds = model(sent_id, masks)
    # pred_label.extend(preds.detach().cpu().numpy())
    # true_label.extend(labels.detach().cpu().numpy())
    pred_label_1.extend(preds[:, 0].detach().cpu().numpy())
    pred_label_2.extend(preds[:, 1].detach().cpu().numpy())
    true_label_1.extend(labels[:, 0].detach().cpu().numpy())
    true_label_2.extend(labels[:, 1].detach().cpu().numpy())

# mae = mean_absolute_error(true_label, pred_label)
mae_1 = mean_absolute_error(true_label_1, pred_label_1)
mae_2 = mean_absolute_error(true_label_2, pred_label_2)

#print(f'MAE: {mae}')

print(f'MAE for Valence: {mae_1}')
print(f'MAE for Arousal: {mae_2}')
#Ckipall2000
#MAE for Valence: 0.9184504151344299
#MAE for Arousal: 1.0241398811340332


# def predict_sentiment(sentence, model, tokenizer):
#     encoded_sent = tokenizer.encode_plus(
#         sentence,
#         truncation=True,
#         max_length=50,
#         add_special_tokens=True,
#         # pad_to_max_length=True,
#         padding='longest',
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     input_ids = encoded_sent['input_ids'].to(device)
#     attention_mask = encoded_sent['attention_mask'].to(device)

#     with torch.no_grad():
#         output = model(input_ids, attention_mask)

#     return output.item()

def predict_sentiment(sentence, model, tokenizer):
    encoded_sent = tokenizer.encode_plus(
        sentence,
        truncation=True,
        max_length=50,
        add_special_tokens=True,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt'
    )
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda'if torch.cuda.is_available() else print("gpu error"))
    input_ids = encoded_sent['input_ids'].to(device)
    attention_mask = encoded_sent['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    return output.cpu().numpy()

