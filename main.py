# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T22:05:42.779436Z","iopub.execute_input":"2023-08-18T22:05:42.779907Z","iopub.status.idle":"2023-08-18T22:05:42.793398Z","shell.execute_reply.started":"2023-08-18T22:05:42.779866Z","shell.execute_reply":"2023-08-18T22:05:42.792319Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T22:05:42.795428Z","iopub.execute_input":"2023-08-18T22:05:42.796055Z","iopub.status.idle":"2023-08-18T22:05:42.806189Z","shell.execute_reply.started":"2023-08-18T22:05:42.795983Z","shell.execute_reply":"2023-08-18T22:05:42.805237Z"}}
import tqdm

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T22:05:42.807749Z","iopub.execute_input":"2023-08-18T22:05:42.808105Z","iopub.status.idle":"2023-08-18T22:05:43.194039Z","shell.execute_reply.started":"2023-08-18T22:05:42.808075Z","shell.execute_reply":"2023-08-18T22:05:43.193043Z"}}
data = pd.read_csv("/kaggle/input/data-tr/data_for_train.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T22:05:43.196980Z","iopub.execute_input":"2023-08-18T22:05:43.197399Z","iopub.status.idle":"2023-08-18T22:05:45.592404Z","shell.execute_reply.started":"2023-08-18T22:05:43.197363Z","shell.execute_reply":"2023-08-18T22:05:45.591342Z"}}
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 读取CSV文件
data.dropna(axis=0, how='any', inplace=True)
data['category'] = data['category'].apply(lambda x: x + 1)
# 设定设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载BERT模型和tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3分类任务
CUDA_LAUNCH_BLOCKING = 1
model.to(device)


# 数据处理
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# 数据加载
max_length = 128
batch_size = 64

train_texts = data['clean_text'].tolist()
train_labels = data['category'].tolist()

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T22:05:45.593960Z","iopub.execute_input":"2023-08-18T22:05:45.594620Z","iopub.status.idle":"2023-08-18T23:11:42.718924Z","shell.execute_reply.started":"2023-08-18T22:05:45.594583Z","shell.execute_reply":"2023-08-18T23:11:42.717965Z"}}
from tqdm import tqdm  # 导入 tqdm 函数

num_epochs = 2
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # 使用 tqdm 包装 train_loader 循环
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as t:
        for batch in t:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            t.set_postfix({"Average Loss": total_loss / (t.n + 1)})  # 更新进度条的后缀信息

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {average_loss:.4f}")

# 保存模型
model.save_pretrained("fine_tuned_bert_model")
tokenizer.save_pretrained("fine_tuned_bert_model")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T23:11:42.720566Z","iopub.execute_input":"2023-08-18T23:11:42.720928Z","iopub.status.idle":"2023-08-18T23:12:09.192485Z","shell.execute_reply.started":"2023-08-18T23:11:42.720895Z","shell.execute_reply":"2023-08-18T23:12:09.191482Z"}}
data_pr = pd.read_excel('/kaggle/input/predict/data_for_predict.csv')
data_pr = data_pr[['Timestamp', 'WebTime', 'Text']]
data_pr.dropna(axis=0, how='any', inplace=True)
data_pr.reset_index(drop=True, inplace=True)

# %% [code]
data

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T23:12:09.194064Z","iopub.execute_input":"2023-08-18T23:12:09.194414Z","iopub.status.idle":"2023-08-18T23:17:28.113972Z","shell.execute_reply.started":"2023-08-18T23:12:09.194377Z","shell.execute_reply":"2023-08-18T23:17:28.112929Z"}}

# Define the DataFrame and column name
data = data_pr
text_column = 'Text'

# Define the max sequence length and batch size
max_length = 128
batch_size = 32


# Create a DataLoader for prediction
def create_prediction_loader(texts, tokenizer, max_length, batch_size):
    dataset = CustomDataset(texts, [0] * len(texts), tokenizer, max_length)  # Dummy labels, not used during prediction
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Create the prediction DataLoader
prediction_loader = create_prediction_loader(data[text_column].tolist(), tokenizer, max_length, batch_size)

# Make predictions
predictions = []
model.eval()

with torch.no_grad(), tqdm(total=len(prediction_loader), desc="Predicting") as t:
    for batch in prediction_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend(predicted_labels)

        t.update(1)

# Add predictions to the DataFrame
data['predictions'] = predictions

data.to_csv('out_come.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T23:19:52.849262Z","iopub.execute_input":"2023-08-18T23:19:52.849679Z","iopub.status.idle":"2023-08-18T23:19:52.855333Z","shell.execute_reply.started":"2023-08-18T23:19:52.849643Z","shell.execute_reply":"2023-08-18T23:19:52.853924Z"}}
from pandas import DataFrame
# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:14:33.828801Z","iopub.execute_input":"2023-08-19T00:14:33.829211Z","iopub.status.idle":"2023-08-19T00:14:33.869060Z","shell.execute_reply.started":"2023-08-19T00:14:33.829176Z","shell.execute_reply":"2023-08-19T00:14:33.867993Z"}}
tdata = DataFrame()
tdata['Value'] = data['predictions'] - 1
tdata['Time'] = data['WebTime'].apply(lambda x: x[:10])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:14:37.046597Z","iopub.execute_input":"2023-08-19T00:14:37.047313Z","iopub.status.idle":"2023-08-19T00:14:37.061817Z","shell.execute_reply.started":"2023-08-19T00:14:37.047277Z","shell.execute_reply":"2023-08-19T00:14:37.060672Z"}}
tdata

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:14:59.792736Z","iopub.execute_input":"2023-08-19T00:14:59.793179Z","iopub.status.idle":"2023-08-19T00:14:59.802471Z","shell.execute_reply.started":"2023-08-19T00:14:59.793147Z","shell.execute_reply":"2023-08-19T00:14:59.801433Z"}}
v_mean = tdata.groupby(tdata['Time']).mean()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:15:01.650691Z","iopub.execute_input":"2023-08-19T00:15:01.651098Z","iopub.status.idle":"2023-08-19T00:15:01.672281Z","shell.execute_reply.started":"2023-08-19T00:15:01.651062Z","shell.execute_reply":"2023-08-19T00:15:01.670966Z"}}
v_mean

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T23:52:32.070574Z","iopub.execute_input":"2023-08-18T23:52:32.070947Z","iopub.status.idle":"2023-08-18T23:52:32.085819Z","shell.execute_reply.started":"2023-08-18T23:52:32.070917Z","shell.execute_reply":"2023-08-18T23:52:32.084485Z"}}
v_mean.to_csv('sentiment_trend.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T23:53:39.914122Z","iopub.execute_input":"2023-08-18T23:53:39.915488Z","iopub.status.idle":"2023-08-18T23:53:39.955127Z","shell.execute_reply.started":"2023-08-18T23:53:39.915443Z","shell.execute_reply":"2023-08-18T23:53:39.953862Z"}}
import pandas as pd
import matplotlib.pyplot as plt

# 将 'Date' 列转换为 datetime 类型
tdata['Time'] = pd.to_datetime(tdata['Time'])

# 设置 'Date' 为索引
tdata.set_index('Time', inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T23:58:06.313412Z","iopub.execute_input":"2023-08-18T23:58:06.313782Z","iopub.status.idle":"2023-08-18T23:58:07.270862Z","shell.execute_reply.started":"2023-08-18T23:58:06.313752Z","shell.execute_reply":"2023-08-18T23:58:07.269800Z"}}


# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(tdata.index, tdata['Value'], marker='o', linestyle='--')
plt.title('value_by_date')
plt.xlabel('Date')
plt.ylabel('Value')

plt.ylim(-2, 2)
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:08:42.115472Z","iopub.execute_input":"2023-08-19T00:08:42.115856Z","iopub.status.idle":"2023-08-19T00:08:43.312341Z","shell.execute_reply.started":"2023-08-19T00:08:42.115824Z","shell.execute_reply":"2023-08-19T00:08:43.311197Z"}}
from textblob import TextBlob
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:14:45.264225Z","iopub.execute_input":"2023-08-19T00:14:45.264640Z","iopub.status.idle":"2023-08-19T00:14:45.277724Z","shell.execute_reply.started":"2023-08-19T00:14:45.264608Z","shell.execute_reply":"2023-08-19T00:14:45.276510Z"}}
tdata.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:15:31.405832Z","iopub.execute_input":"2023-08-19T00:15:31.406254Z","iopub.status.idle":"2023-08-19T00:15:31.431321Z","shell.execute_reply.started":"2023-08-19T00:15:31.406221Z","shell.execute_reply":"2023-08-19T00:15:31.429927Z"}}
tdata['Time'] = pd.to_datetime(tdata['Time'])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:42:14.421867Z","iopub.execute_input":"2023-08-19T00:42:14.422252Z","iopub.status.idle":"2023-08-19T00:42:14.527923Z","shell.execute_reply.started":"2023-08-19T00:42:14.422221Z","shell.execute_reply":"2023-08-19T00:42:14.526864Z"}}
import pandas as pd
from textblob import TextBlob
from statsmodels.tsa.ar_model import AutoReg


# 定义一个函数进行AR拟合并返回结果摘要
def fit_ar(data):
    model = AutoReg(data, lags=1)
    model_fit = model.fit()
    return model_fit.summary()


# 将数据划分为2023年的四个季度
quarters = {

    "Q1": ('2016-06-01', '2019-10-01'),
    "Q2": ('2019-10-02', '2020-01-30'),
    "Q3": ('2020-01-31', '2022-02-23'),
    "Q4": ('2022-02-24', '2023-08-11')
}

# 对每个季度的数据使用AR模型进行拟合并打印结果
for q, (start_date, end_date) in quarters.items():
    subset = tdata[(tdata['Time'] >= start_date) & (tdata['Time'] <= end_date)]['Value']
    print(f"AR Fit for {q} of 2023:")
    print(fit_ar(subset))
    print("----------\n")


# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:44:33.760000Z","iopub.execute_input":"2023-08-19T00:44:33.760403Z","iopub.status.idle":"2023-08-19T00:44:36.369562Z","shell.execute_reply.started":"2023-08-19T00:44:33.760373Z","shell.execute_reply":"2023-08-19T00:44:36.367106Z"}}
def fit_and_predict_ar(data):
    model = AutoReg(data, lags=1)
    model_fit = model.fit()
    return model_fit.predict(start=1, end=len(data))


# 创建一个图形，其中有4个子图，每个子图表示一个季度
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for ax, (q, (start_date, end_date)) in zip(axes, quarters.items()):
    subset = tdata[(tdata['Time'] >= start_date) & (tdata['Time'] <= end_date)]

    # 绘制实际的情感指数
    ax.plot(subset['Time'], subset['Value'], label="Actual", color="blue")

    # 使用AR模型进行拟合并预测
    prediction = fit_and_predict_ar(subset['Value'])
    ax.plot(subset['Time'][0:], prediction, label="Predicted", linestyle="--", color="red")

    ax.set_title(f"Sentiment Index and AR Prediction for {q} 2023")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Index')
    ax.legend()

plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:15:32.937125Z","iopub.execute_input":"2023-08-19T00:15:32.937500Z","iopub.status.idle":"2023-08-19T00:15:50.223806Z","shell.execute_reply.started":"2023-08-19T00:15:32.937471Z","shell.execute_reply":"2023-08-19T00:15:50.222766Z"}}
# 检查时间序列的稳定性 (Augmented Dickey-Fuller test)
result = adfuller(tdata['Value'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# 如果 p-value > 0.05，你可能需要差分操作来使时间序列稳定

# AR 模型
model = AutoReg(tdata['Value'], lags=1)
model_fit = model.fit()

# 预测
tdata['Prediction'] = model_fit.predict(start=1, end=len(tdata))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(tdata['Time'], tdata['Value'], marker='o', linestyle='-', label="Actual")
plt.plot(tdata['Time'], tdata['Prediction'], marker='x', linestyle='--', color="red", label="Predicted")
plt.title('Sentiment Index and AR Prediction')
plt.xlabel('Date')
plt.ylabel('Sentiment Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T00:27:05.225829Z","iopub.execute_input":"2023-08-19T00:27:05.226363Z","iopub.status.idle":"2023-08-19T00:27:05.250685Z","shell.execute_reply.started":"2023-08-19T00:27:05.226298Z","shell.execute_reply":"2023-08-19T00:27:05.249547Z"}}
# 打印回归的拟合结果
print(model_fit.summary())

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:17:13.399918Z","iopub.execute_input":"2023-08-19T01:17:13.400313Z","iopub.status.idle":"2023-08-19T01:17:13.406379Z","shell.execute_reply.started":"2023-08-19T01:17:13.400282Z","shell.execute_reply":"2023-08-19T01:17:13.405205Z"}}
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:17:15.265209Z","iopub.execute_input":"2023-08-19T01:17:15.266053Z","iopub.status.idle":"2023-08-19T01:17:15.298213Z","shell.execute_reply.started":"2023-08-19T01:17:15.265980Z","shell.execute_reply":"2023-08-19T01:17:15.297064Z"}}
# 将 'Date' 列转换为 datetime 类型
tdata['Time'] = pd.to_datetime(tdata['Time'])

# 设置 'Date' 为索引
tdata.set_index('Time', inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:25:10.348355Z","iopub.execute_input":"2023-08-19T01:25:10.348752Z","iopub.status.idle":"2023-08-19T01:25:10.376308Z","shell.execute_reply.started":"2023-08-19T01:25:10.348719Z","shell.execute_reply":"2023-08-19T01:25:10.375111Z"}}
# 计算每年的情感指数的平均值
annual_sentiment = tdata.resample('M').mean()

print(annual_sentiment)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:25:42.263408Z","iopub.execute_input":"2023-08-19T01:25:42.263896Z","iopub.status.idle":"2023-08-19T01:25:42.280235Z","shell.execute_reply.started":"2023-08-19T01:25:42.263859Z","shell.execute_reply":"2023-08-19T01:25:42.278944Z"}}
annual_sentiment.to_csv('sentiment_trend_by_month.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:47:36.968702Z","iopub.execute_input":"2023-08-19T01:47:36.969106Z","iopub.status.idle":"2023-08-19T01:47:36.978928Z","shell.execute_reply.started":"2023-08-19T01:47:36.969070Z","shell.execute_reply":"2023-08-19T01:47:36.977642Z"}}
df = pd.read_csv("/kaggle/input/inflationbymontion/inflation_by_month.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:47:38.251233Z","iopub.execute_input":"2023-08-19T01:47:38.251671Z","iopub.status.idle":"2023-08-19T01:47:38.265543Z","shell.execute_reply.started":"2023-08-19T01:47:38.251636Z","shell.execute_reply":"2023-08-19T01:47:38.264379Z"}}
# 将 'Date' 列转换为 datetime 类型
df['Time'] = pd.to_datetime(df['Time'])

# 设置 'Date' 为索引
df.set_index('Time', inplace=True)
df.index.freq = 'M'

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:47:40.472543Z","iopub.execute_input":"2023-08-19T01:47:40.472929Z","iopub.status.idle":"2023-08-19T01:47:40.927741Z","shell.execute_reply.started":"2023-08-19T01:47:40.472897Z","shell.execute_reply":"2023-08-19T01:47:40.926554Z"}}

# 使用AR(2)模型并加入外生变量来拟合数据
model = AutoReg(df['inflation'], lags=2, exog=df['Value'])
results = model.fit()

# 输出模型摘要
print(results.summary())

# 预测和绘图
forecast = results.predict(start=len(df), end=len(df) + 20, exog_oos=df['Value'].iloc[-21:].values.reshape(-1, 1))
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['inflation'], label="Actual Inflation")
plt.plot(pd.date_range(start=df.index[-1], periods=21, freq='D')[0:], forecast, label="Forecasted Inflation",
         linestyle="--")
plt.legend()
plt.title("AR(2) Model with Sentiment to Predict Inflation")
plt.show()

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T01:46:34.849541Z","iopub.execute_input":"2023-08-19T01:46:34.850091Z","iopub.status.idle":"2023-08-19T01:46:35.614011Z","shell.execute_reply.started":"2023-08-19T01:46:34.850054Z","shell.execute_reply":"2023-08-19T01:46:35.609101Z"}}

# 使用AR(2)模型并加入外生变量来拟合数据
model = AutoReg(df['inflation'], lags=2, exog=df['Value'])
results = model.fit()

# 输出模型摘要
print(results.summary())

# 预测和绘图
forecast = results.predict(start=len(df), end=len(df) + 20, exog_oos=df['Prediction'].iloc[-21:].values.reshape(-1, 1))
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['inflation'], label="Actual Inflation")
plt.plot(pd.date_range(start=df.index[-1], periods=21, freq='D')[0:], forecast, label="Forecasted Inflation",
         linestyle="--")
plt.legend()
plt.title("AR(2) Model with Sentiment to Predict Inflation")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T02:16:17.657921Z","iopub.execute_input":"2023-08-19T02:16:17.658362Z","iopub.status.idle":"2023-08-19T02:16:17.674840Z","shell.execute_reply.started":"2023-08-19T02:16:17.658328Z","shell.execute_reply":"2023-08-19T02:16:17.673643Z"}}
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# 假设df是你的数据，其中'sentiment'是情绪指数，'inflation'是通货膨胀指数
df = pd.read_csv('/kaggle/input/inflationbymonths/inflation_by_month.csv')
# 将 'Date' 列转换为 datetime 类型
df['Time'] = pd.to_datetime(df['Time'])

# 设置 'Date' 为索引
df.set_index('Time', inplace=True)
df.index.freq = 'M'

# 取负值
df['Value'] = -df['Value']
df['Prediction'] = -df['Prediction']

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T02:16:20.446819Z","iopub.execute_input":"2023-08-19T02:16:20.447252Z","iopub.status.idle":"2023-08-19T02:16:20.463229Z","shell.execute_reply.started":"2023-08-19T02:16:20.447219Z","shell.execute_reply":"2023-08-19T02:16:20.462047Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T02:22:32.556183Z","iopub.execute_input":"2023-08-19T02:22:32.556555Z","iopub.status.idle":"2023-08-19T02:22:33.891278Z","shell.execute_reply.started":"2023-08-19T02:22:32.556524Z","shell.execute_reply":"2023-08-19T02:22:33.890205Z"}}
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# 分割数据
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 使用情绪指数作为外部变量进行AR(2)模型拟合
model = AutoReg(train['inflation'], lags=2, exog=train['Value'], old_names=False)
results = model.fit()

# 使用测试集的情绪指数进行预测
predictions = results.predict(start=len(train), end=len(train) + len(test) - 1, exog_oos=test['Value'])

# 计算 MSE, RMSE, MAE
mse = mean_squared_error(test['inflation'], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test['inflation'], predictions)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# R-squared
model = sm.OLS(test['inflation'], sm.add_constant(predictions)).fit()
print(f"R2: {model.rsquared}")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-19T02:16:25.322515Z","iopub.execute_input":"2023-08-19T02:16:25.322902Z","iopub.status.idle":"2023-08-19T02:16:25.864716Z","shell.execute_reply.started":"2023-08-19T02:16:25.322872Z","shell.execute_reply":"2023-08-19T02:16:25.863724Z"}}
# 绘制真实值
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['inflation'], label='True', color='blue')

# 绘制预测值
plt.plot(test.index, predictions, label='Prediction', color='red', linestyle='--')

plt.title('Inflation Index: True vs Predicted')
plt.xlabel('Date')
plt.ylabel('Inflation Index')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
