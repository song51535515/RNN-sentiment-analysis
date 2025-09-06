# 基于RNN的影评情感分析

本项目实现了一套对于《流浪地球》猫眼影评完整的情感分析方案，涵盖**数据获取**、**预处理**、**模型构建**到**情感分类**的全流程。将网络爬取的评论数据进行初步清洗，通过深度学习模型（RNN）实现评论情感的三分类（好评 / 中评 / 差评）。

## 核心功能

1.  **评论数据爬取**：从网络批量获取带评分的用户评论
2.  **数据预处理**：清洗、分词及情感标签转换
3.  **深度学习模型**：基于 RNN 的情感分类模型
4.  **模型评估**：自动计算训练集与测试集的分类准确率

## 项目结构



```plaintext
.
├── gatdata.ipynb         # 评论数据爬取与预处理脚本
├── rnn_model.ipynb       # RNN情感分类模型实现
├── alldata/              # 训练集与测试集数据
│   ├── train-all.csv
│   └── test-all.csv
├── comments_new.csv      # 爬取的原始评论数据
├── word2vec_model        # 预训练的词向量模型
└── RNN.parameters  
```


## 模块详解

### 1. 数据爬取与预处理（getdata.ipynb）

#### 功能说明

-   从网络 API 批量爬取用户评论数据（支持时间范围筛选）
-   对原始评论进行清洗、分词和情感标签转换
-   将处理后的数据保存为 CSV 格式，用于模型训练
#### 核心函数
```python
# 爬取API数据
def requestApi(url):
    # 带请求头的HTTP请求实现

# 解析评论数据
def getData(html):
    # 提取评论者、城市、内容、评分、时间等信息

# 保存数据到CSV
def saveData(comments, is_first):
    # 支持表头控制的CSV写入

# 爬虫主函数
def main():
    # 时间范围控制与批量爬取逻辑

```

爬取结果csv文件如图所示

![输入图片说明](/img/1.jpg)

dataframe输出

![输入图片说明](/img/2.jpg)
#### 数据清洗

1.  爬取数据格式：包含`nickName`(用户名)、`cityName`(城市)、`content`(评论内容)、`score`(原始评分)、`startTime`(评论时间)
2.  预处理后格式：转换为`content`(原始评论)、`score`(三分类标签：1 - 差 / 2 - 中 / 3 - 好)、`content_cut`(分词结果)，代码如图所示
```python
import pandas as pd
import numpy as np
#!pip3 install jieba
import jieba

data = pd.read_csv('./comments_new.csv').astype(str)
# , names=['Name', 'Area', 'comment', 'star', 'time'])
data['score'] = data['score'].replace(regex=True, inplace=False, to_replace=['nan'], value='')
data1 = data[~data['score'].isin(['0', ' '])]
data1['score'] = pd.to_numeric(data1['score'], errors='coerce')
data1['score'] = data1['score'].apply(
    lambda x: '1' if x in [0.5,1,1.5,2]  # 1、2 → 低
              else '2' if x in [2.5,3,3.5]  # 3 → 中
              else '3'             # 4、5 → 高
)
# data1['score'] = data1['score'].map(type_dict)
data1['cut'] = data1["content"].apply(lambda x: ' '.join(jieba.cut(x)))

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 将短评中的停用词删去
def sentence_div(text):
    # 将短评按空格划分成单词并形成列表
    sentence = text.strip().split()
    # 加载停用词的路径
    stopwords = stopwordslist(r'cn-stopwords.txt')
    # 创建一个空字符串
    outstr = ''
    # 遍历短评列表中每个单词
    for word in sentence:
        if word not in stopwords:  # 判断词汇是否在停用词表里
            if len(word) >= 1:  # 单词长度要大于1
                if word != '\t':  # 单词不能为tab
                    if word not in outstr:  # 去重: 如果单词在outstr中则不加入
                        outstr += ' '  # 分割
                        outstr += word  # 将词汇加入outstr
    # 返回字符串
    return outstr

data1['content_cut'] = data1['cut'].apply(sentence_div)
data1 = data1[['content', 'score', 'content_cut']]
data1.to_csv("./all_comments.csv", index=None)
```

清洗后的数据
![输入图片说明](/img/3.jpg)


### 2. RNN 情感分类模型(rnn_model.ipynb)

#### 模型概述

采用循环神经网络 (RNN) 实现情感三分类，通过 Word2Vec 进行词向量表示，捕获文本序列特征实现情感判断。

#### 核心流程

1.  **数据读取与预处理**

```python
# 读取训练集和测试集
def read_comments(train_file, test_file)

# 生成词语tokens
def create_tokens(train_array, test_array)
```
2.  **词向量生成**    
```python
    # 使用Word2Vec生成词向量
    def word_vec(tokens):
        model = Word2Vec(tokens, sg=0, vector_size=300, window=5, min_count=1, epochs=7, negative=10)
```
     
3.  **模型构建**

RNN的**核心公式**如下。

隐状态 $H$:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$

输出结果 $O$:

$$O_{t} = H_{t}W_{hq} + b_q$$

其中，添加了 ${H}_{t-1}$ 代表上个时序**隐状态**，${W}_{hh}$ 代表了其对应的**权重矩阵**, $O_{t}$代表**时间段t的输出**。

图示如下:
![输入图片说明](/img/4.jpg)
由此可见，要想使用 RNN 网络预测得到结果，我们大体上需要两个部分组成：**RNN层(生成最终隐状态 $H$ )、Linear全连接层(生成结果 $O$)**。

但是又需要将所有评论语句变为向量投入到网络中，所以还需要一部分**Embedding词嵌入**模型，用于将所有的评论信息转化为矩阵信息，所以共需要三部分构成。下面来定义RNN模型：

   ```python
    class RNNModel(nn.Module):
        def __init__(self, id_token_voc, embedding_dim, hidden_dim, output_dim, vectors):
            self.embedding = nn.Embedding(len(id_token_voc), embedding_dim)  # 词嵌入层
            self.rnn = nn.RNN(embedding_dim, hidden_dim)  # RNN层
            self.linear = nn.Linear(hidden_dim, output_dim)  # 全连接层
    
        def forward(self, X):
            # 前向传播逻辑
   ```
    
          
4.  **训练与评估**

梯度裁剪，是对梯度进行限制，防止出现**梯度爆炸**的情况，以免影响模型训练。

具体的裁剪方法如下公式所示:

$$g \leftarrow min(1, \frac{\theta}{||g||})g$$

其中， $||g||$代表**梯度的二范数**, $\theta$ 代表**设定范围**。
     
   ```python
    # 梯度裁剪（防止梯度爆炸）
    def grad_clipping(net, theta)
    
    # 模型评估
    def evaluate_net(net, train_iter, test_iter, device)
    
    # 模型训练
    def train(net, train_iter, test_iter, loss, updater, num_epochs, device)
   ```
最终模型的部分运行结果如图所示
![输入图片说明](/img/图片6.png)

用 classification_report() 函数得到RNN模型每个类别的预
测**准确率**，**精确率**，**召回率**以及**F1分数**的各项指标值。
![输入图片说明](/img/图片1.png)

混淆矩阵

![输入图片说明](/img/图片2.png)

容易发现，总评价准确率=类别3占总测试集比例，且混淆矩阵第一第二列均为0，这意味着模型并没有将任何语料识别为1或2，**无法断定模型具有识别类别1,2的能力**，只将所有数据都预测为了类别3。
这是由于类别3占样本总数比例过高导致的，为此调整训练集和测试集数据量，**使三个类别数据的预测准确率均不为0**，即模型对三个类别都有预测能力，得出模型**真实准确率约为70％左右**
![输入图片说明](/img/图片3.png)
    
    
    
      
    
      
    




