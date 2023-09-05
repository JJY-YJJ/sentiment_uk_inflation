# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:33:33.012679Z","iopub.execute_input":"2023-08-18T13:33:33.013053Z","iopub.status.idle":"2023-08-18T13:33:33.052234Z","shell.execute_reply.started":"2023-08-18T13:33:33.013017Z","shell.execute_reply":"2023-08-18T13:33:33.051341Z"}}
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:33:33.053529Z","iopub.execute_input":"2023-08-18T13:33:33.053783Z","iopub.status.idle":"2023-08-18T13:33:33.058337Z","shell.execute_reply.started":"2023-08-18T13:33:33.053760Z","shell.execute_reply":"2023-08-18T13:33:33.057348Z"}}
# pip install "numpy>=1.16.5,<1.23.0"

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:33:33.059751Z","iopub.execute_input":"2023-08-18T13:33:33.060075Z","iopub.status.idle":"2023-08-18T13:33:47.091325Z","shell.execute_reply.started":"2023-08-18T13:33:33.060045Z","shell.execute_reply":"2023-08-18T13:33:47.089439Z"}}
pip install Cython

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:33:47.094494Z","iopub.execute_input":"2023-08-18T13:33:47.094877Z","iopub.status.idle":"2023-08-18T13:34:05.686298Z","shell.execute_reply.started":"2023-08-18T13:33:47.094828Z","shell.execute_reply":"2023-08-18T13:34:05.685000Z"}}
pip install numpy==1.23.2

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:05.688335Z","iopub.execute_input":"2023-08-18T13:34:05.688719Z","iopub.status.idle":"2023-08-18T13:34:19.045900Z","shell.execute_reply.started":"2023-08-18T13:34:05.688681Z","shell.execute_reply":"2023-08-18T13:34:19.044811Z"}}
#general purpose packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#data processing
import re, string
import emoji
import nltk

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

#keras
import tensorflow as tf
from tensorflow import keras


#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Set seed for reproducibility
import random
seed_value = 2042
random.seed(seed_value)
#set seed for reproducibility
seed=42

#set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)


# PyTorch LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# Transformers library for BERT
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import time


# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:19.047386Z","iopub.execute_input":"2023-08-18T13:34:19.048563Z","iopub.status.idle":"2023-08-18T13:34:20.466688Z","shell.execute_reply.started":"2023-08-18T13:34:19.048521Z","shell.execute_reply":"2023-08-18T13:34:20.465726Z"}}
df_visualisation = pd.read_csv('../input/16-23sentiment/inflation_twitter.csv',encoding='ISO-8859-1',dtype='unicode')
df = pd.read_csv('../input/sentimentdata/Twitter_Data.csv',encoding='ISO-8859-1',dtype='unicode')
# df_test = pd.read_csv('../input/covid-19-nlp-df-classification/Corona_NLP_test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:20.468302Z","iopub.execute_input":"2023-08-18T13:34:20.468665Z","iopub.status.idle":"2023-08-18T13:34:20.493310Z","shell.execute_reply.started":"2023-08-18T13:34:20.468631Z","shell.execute_reply":"2023-08-18T13:34:20.492270Z"}}
df_visualisation.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:20.494687Z","iopub.execute_input":"2023-08-18T13:34:20.495115Z","iopub.status.idle":"2023-08-18T13:34:20.506102Z","shell.execute_reply.started":"2023-08-18T13:34:20.495082Z","shell.execute_reply":"2023-08-18T13:34:20.504903Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:20.507532Z","iopub.execute_input":"2023-08-18T13:34:20.508091Z","iopub.status.idle":"2023-08-18T13:34:20.514787Z","shell.execute_reply.started":"2023-08-18T13:34:20.508058Z","shell.execute_reply":"2023-08-18T13:34:20.513618Z"}}
#数据内容和数据可视化

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:20.520133Z","iopub.execute_input":"2023-08-18T13:34:20.520477Z","iopub.status.idle":"2023-08-18T13:34:20.824147Z","shell.execute_reply.started":"2023-08-18T13:34:20.520452Z","shell.execute_reply":"2023-08-18T13:34:20.823058Z"}}
df_visualisation.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:20.825724Z","iopub.execute_input":"2023-08-18T13:34:20.826103Z","iopub.status.idle":"2023-08-18T13:34:20.864441Z","shell.execute_reply.started":"2023-08-18T13:34:20.826069Z","shell.execute_reply":"2023-08-18T13:34:20.863551Z"}}
df_visualisation['WebTime'] = pd.to_datetime(df_visualisation['WebTime'])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:20.865683Z","iopub.execute_input":"2023-08-18T13:34:20.866109Z","iopub.status.idle":"2023-08-18T13:34:21.080786Z","shell.execute_reply.started":"2023-08-18T13:34:20.866076Z","shell.execute_reply":"2023-08-18T13:34:21.079744Z"}}
np.sum(df_visualisation.duplicated()) 

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:21.082133Z","iopub.execute_input":"2023-08-18T13:34:21.082561Z","iopub.status.idle":"2023-08-18T13:34:21.356237Z","shell.execute_reply.started":"2023-08-18T13:34:21.082527Z","shell.execute_reply":"2023-08-18T13:34:21.355216Z"}}
df_visualisation.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:21.357589Z","iopub.execute_input":"2023-08-18T13:34:21.358044Z","iopub.status.idle":"2023-08-18T13:34:21.362903Z","shell.execute_reply.started":"2023-08-18T13:34:21.358009Z","shell.execute_reply":"2023-08-18T13:34:21.361946Z"}}
 filepath = '/kaggle/working'

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:21.364159Z","iopub.execute_input":"2023-08-18T13:34:21.365159Z","iopub.status.idle":"2023-08-18T13:34:21.372611Z","shell.execute_reply.started":"2023-08-18T13:34:21.365126Z","shell.execute_reply":"2023-08-18T13:34:21.371712Z"}}
#tweets by date 

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:21.373915Z","iopub.execute_input":"2023-08-18T13:34:21.374389Z","iopub.status.idle":"2023-08-18T13:34:21.741814Z","shell.execute_reply.started":"2023-08-18T13:34:21.374357Z","shell.execute_reply":"2023-08-18T13:34:21.740875Z"}}
tweets_per_day = df_visualisation['WebTime'].dt.strftime('%m-%d').value_counts().sort_index().reset_index(name='counts')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:21.743090Z","iopub.execute_input":"2023-08-18T13:34:21.743417Z","iopub.status.idle":"2023-08-18T13:34:33.879070Z","shell.execute_reply.started":"2023-08-18T13:34:21.743385Z","shell.execute_reply":"2023-08-18T13:34:33.877797Z"}}
plt.figure(figsize=(20,5))
# sns.distplot(irisDf['petal_length'],kde=True,rug=True) #kde密度曲线 rug边际毛毯
ax = sns.lineplot(x='index', y='counts', data=tweets_per_day,errorbar=('ci', False), palette='Blues_r')
plt.title('Tweets count by date')
plt.yticks([])
# ax.line_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()
# fig_path为想要存入的文件夹或地址
plt_name = 'tweets_by_date.png'
plt_path = filepath + '/' + plt_name
line_fig = ax.get_figure()
line_fig.savefig(plt_path, dpi = 400)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:33.880568Z","iopub.execute_input":"2023-08-18T13:34:33.881043Z","iopub.status.idle":"2023-08-18T13:34:33.885770Z","shell.execute_reply.started":"2023-08-18T13:34:33.881008Z","shell.execute_reply":"2023-08-18T13:34:33.884896Z"}}
#tweets by keywords

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:33.887500Z","iopub.execute_input":"2023-08-18T13:34:33.888220Z","iopub.status.idle":"2023-08-18T13:34:33.977919Z","shell.execute_reply.started":"2023-08-18T13:34:33.888180Z","shell.execute_reply":"2023-08-18T13:34:33.976891Z"}}
tweets_per_keyword = df_visualisation['re_keyword'].value_counts().reset_index(name='keyword_counts')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:33.981771Z","iopub.execute_input":"2023-08-18T13:34:33.982331Z","iopub.status.idle":"2023-08-18T13:34:34.466747Z","shell.execute_reply.started":"2023-08-18T13:34:33.982296Z","shell.execute_reply":"2023-08-18T13:34:34.465804Z"}}
plt.figure(figsize=(20,5))
ax = sns.barplot(x='index', y='keyword_counts', data=tweets_per_keyword,edgecolor = 'black',ci=False, palette='Blues_r')
plt.title('Tweets count by keyword')
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:34.468036Z","iopub.execute_input":"2023-08-18T13:34:34.468388Z","iopub.status.idle":"2023-08-18T13:34:34.475906Z","shell.execute_reply.started":"2023-08-18T13:34:34.468355Z","shell.execute_reply":"2023-08-18T13:34:34.475014Z"}}
#数据处理

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:34.477507Z","iopub.execute_input":"2023-08-18T13:34:34.477931Z","iopub.status.idle":"2023-08-18T13:34:34.484588Z","shell.execute_reply.started":"2023-08-18T13:34:34.477898Z","shell.execute_reply":"2023-08-18T13:34:34.483668Z"}}
#去重复值

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:34.485958Z","iopub.execute_input":"2023-08-18T13:34:34.486367Z","iopub.status.idle":"2023-08-18T13:34:34.799187Z","shell.execute_reply.started":"2023-08-18T13:34:34.486335Z","shell.execute_reply":"2023-08-18T13:34:34.798041Z"}}
data = pd.read_csv('../input/sentimentdata/Twitter_Data.csv',encoding='ISO-8859-1',dtype='unicode')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:34.800657Z","iopub.execute_input":"2023-08-18T13:34:34.801012Z","iopub.status.idle":"2023-08-18T13:34:34.902859Z","shell.execute_reply.started":"2023-08-18T13:34:34.800978Z","shell.execute_reply":"2023-08-18T13:34:34.901951Z"}}
np.sum(data.duplicated()) 

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:34.904117Z","iopub.execute_input":"2023-08-18T13:34:34.904477Z","iopub.status.idle":"2023-08-18T13:34:35.012826Z","shell.execute_reply.started":"2023-08-18T13:34:34.904443Z","shell.execute_reply":"2023-08-18T13:34:35.011889Z"}}
# 我们把相同的丢弃
df = data.drop_duplicates()
df.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:35.014340Z","iopub.execute_input":"2023-08-18T13:34:35.014713Z","iopub.status.idle":"2023-08-18T13:34:35.117586Z","shell.execute_reply.started":"2023-08-18T13:34:35.014679Z","shell.execute_reply":"2023-08-18T13:34:35.116416Z"}}
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:35.119303Z","iopub.execute_input":"2023-08-18T13:34:35.119668Z","iopub.status.idle":"2023-08-18T13:34:35.159303Z","shell.execute_reply.started":"2023-08-18T13:34:35.119635Z","shell.execute_reply":"2023-08-18T13:34:35.158154Z"}}
# 检查数据的完整性
print(f"{np.sum(df['clean_text'].isna())} rows have no clean_text")
print(f"{np.sum(df['category'].isna())} rows have no category")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:35.166628Z","iopub.execute_input":"2023-08-18T13:34:35.166923Z","iopub.status.idle":"2023-08-18T13:34:35.173384Z","shell.execute_reply.started":"2023-08-18T13:34:35.166897Z","shell.execute_reply":"2023-08-18T13:34:35.172384Z"}}
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:35.174911Z","iopub.execute_input":"2023-08-18T13:34:35.175545Z","iopub.status.idle":"2023-08-18T13:34:42.580471Z","shell.execute_reply.started":"2023-08-18T13:34:35.175509Z","shell.execute_reply":"2023-08-18T13:34:42.578872Z"}}
df['clean_text'] = df['clean_text'].apply(lambda x:clean_text(x))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:42.581765Z","iopub.execute_input":"2023-08-18T13:34:42.582439Z","iopub.status.idle":"2023-08-18T13:34:42.691562Z","shell.execute_reply.started":"2023-08-18T13:34:42.582388Z","shell.execute_reply":"2023-08-18T13:34:42.690353Z"}}
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-18T13:34:42.693113Z","iopub.execute_input":"2023-08-18T13:34:42.694131Z","iopub.status.idle":"2023-08-18T13:34:42.699182Z","shell.execute_reply.started":"2023-08-18T13:34:42.694092Z","shell.execute_reply":"2023-08-18T13:34:42.698082Z"}}
#bert classification