# Sentiment Analysis of Movie Reviews Based on RNN

This project implements a complete sentiment analysis pipeline for The Wandering Earth reviews from Maoyan, covering the full process from **data acquisition**, **preprocessing**, **model construction**, to **sentiment classification**. The collected reviews are cleaned and processed, and a deep learning model (RNN) is applied to achieve three-class sentiment classification (positive/neutral/negative).

## Key Features
- **Review Data Crawling**: Batch collection of user reviews with ratings from the web  
- **Data Preprocessing**: Cleaning, tokenization, and sentiment label conversion  
- **Deep Learning Model**: RNN-based sentiment classification  
- **Model Evaluation**: Automatic calculation of classification accuracy on training and test sets  

## Project Structure
```
├── gatdata.ipynb         # Script for review data crawling and preprocessing
├── rnn_model.ipynb       # RNN sentiment classification model
├── alldata/              # Training and testing datasets
│   ├── train-all.csv
│   └── test-all.csv
├── comments_new.csv      # Raw crawled review data
├── word2vec_model        # Pretrained word vector model
└── RNN.parameters
```
## Module Details
### Part 1: Data Crawling and Preprocessing (getdata.ipynb)
#### Function Description
- Batch crawls user reviews from the web API (supports time-range filtering)
- Cleans raw reviews, tokenizes, and converts ratings to sentiment labels
- Saves processed data into CSV format for training
#### Core Functions
```python
# Request API data
get_page_content(page):
    # HTTP request with headers

# Save data to CSV
save_to_csv(data, is_first_page=False):
    # CSV writing

# Number of pages to crawl
total_pages = 4  # Can be modified as needed

# Store all crawling results
all_results = {
    'name': [],
    'likes': [],
    'content': [],
    'score': [],
    'time': []
}

for page in range(total_pages):
    print(f"Crawling page {page+1}...")
    
    # Get current page content
    html_content = get_page_content(page)
    if not html_content:
        continue
    
    # Parse page content
    soup = BeautifulSoup(html_content, 'lxml')

    # Extract username - handle possible None values
    list_name = [a.string if a and a.string else '' for a in soup.select('.comment .comment-info a')]

    # Extract number of likes
    list_like = [span.string if span and span.string else '0' for span in soup.select('.comment .comment-vote span')]

    # Extract comment time - check existence before calling strip()
    list_time = []
    for span in soup.select('.comment .comment-info .comment-time'):
        if span and span.string:
            list_time.append(span.string.strip())
        else:
            list_time.append('')

    list_rating = []
    for span in soup.select('span[class*="allstar"][class*="rating"]'):
        # Extract rating value from class
        if span and 'class' in span.attrs:
            class_str = ' '.join(span['class'])
            for part in class_str.split():
                if part.startswith('allstar'):
                    rating = part.replace('allstar', '')
                    list_rating.append(str(int(rating)/10))
                    break
        else:
            list_rating.append('No rating')

    # Extract comment content
    list_data = [span.string if span and span.string else '' for span in soup.select('.comment-content .short')]
    
    # Ensure rating list length matches other lists
    while len(list_rating) < len(list_name):
        list_rating.append('No rating')
    
    # Add current page data to all results
    all_results['name'].extend(list_name)
    all_results['likes'].extend(list_like)
    all_results['content'].extend(list_data)
    all_results['score'].extend(list_rating)
    all_results['time'].extend(list_time)
    
    # Prepare current page data for CSV writing
    page_data = []
    for i in range(len(list_name)):
        page_data.append({
            'name': list_name[i],
            'likes': list_like[i],
            'content': list_data[i],
            'score': list_rating[i],
            'time': list_time[i]
        })
    
    # Write to CSV file, only write header for first page
    save_to_csv(page_data, page == 0)
    
    print(f"Page {page+1} crawling completed, obtained {len(list_name)} comments(s), already written to CSV file\n")

# Print total results statistics
print(f"All pages crawling completed, total {len(all_results['name'])} comment(s) obtained")
print("Results saved to 'comments_new.csv' file")
```
The resulting CSV file looks like this:

![输入图片说明](/img/1.jpg)

DataFrame output:

![输入图片说明](/img/2.jpg)

#### Data Cleaning
- Raw data format: name (username), like count (city), content (review text), score (rating), time (review time)
- Processed format: content (raw review), score (three-class sentiment: 1 - Negative / 2 - Neutral / 3 - Positive), content_cut (tokenized result)

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
    lambda x: '1' if x in [0.5,1,1.5,2]  # 1-2 → negative
              else '2' if x in [2.5,3,3.5]  # 2.5-3.5 → neutral
              else '3'             # 4-5 → positive
)
# data1['score'] = data1['score'].map(type_dict)
data1['cut'] = data1["content"].apply(lambda x: ' '.join(jieba.cut(x)))

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# Delete the stop words in comments
def sentence_div(text):
    # Divide the essay into words according to the spaces and form a list
    sentence = text.strip().split()
    # Path to load stop words
    stopwords = stopwordslist(r'cn-stopwords.txt')
    
    outstr = ''
    # Traverse each word in the short comment list
    for word in sentence:
        if word not in stopwords:  
            if len(word) >= 1:  
                if word != '\t':  
                    if word not in outstr:  
                        outstr += ' '  
                        outstr += word  
   
    return outstr

data1['content_cut'] = data1['cut'].apply(sentence_div)
data1 = data1[['content', 'score', 'content_cut']]
data1.to_csv("./all_comments.csv", index=None)
```

The data after cleaning
![输入图片说明](/img/3.jpg)


### Part 2. RNN Sentiment Classification Model(rnn_model.ipynb)

#### Model Description

To implement three-class sentiment classification using Recurrent Neural Networks (RNN), we utilize Word2Vec for word vector representation to capture text sequence features and achieve sentiment judgment.


#### Data Reading and Preprocessing

```python
# Read training set and test set
def read_comments(train_file, test_file)

# Generate word tokens
def create_tokens(train_array, test_array)
```
#### Generate Word Vector    
```python
    # Word2vec to generate word vectors
    def word_vec(tokens):
        model = Word2Vec(tokens, sg=0, vector_size=300, window=5, min_count=1, epochs=7, negative=10)
```
     
#### Model Construction

The formula of RNN is as follows.

Hidden state $H$:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$

Output $O$:

$$O_{t} = H_{t}W_{hq} + b_q$$

Added $H_{t-1}$ represents the previous time sequence **hidden state**, $W_{hh}$ represents its corresponding **weight matrix**, and $o_t$ represents the output of **time period $t$**.
As follows:
![输入图片说明](/img/4.jpg)

It can be seen that in order to use RNN networks to predict results, we generally need two parts: **RNN layer (generating the final hidden state $H$ ), and Linear fully connected layer (generating the result $O$)**.

However, it is necessary to convert all comment statements into vectors and input them into the network, so a part of the **Embedding Word Embedding Model** is also needed to convert all comment information into matrix information. Therefore, a total of three parts are required. Let's define the RNN model below:

   ```python
    class RNNModel(nn.Module):
        def __init__(self, id_token_voc, embedding_dim, hidden_dim, output_dim, vectors):
            self.embedding = nn.Embedding(len(id_token_voc), embedding_dim)  # Embedding layer
            self.rnn = nn.RNN(embedding_dim, hidden_dim)  # RNN layer
            self.linear = nn.Linear(hidden_dim, output_dim)  # Fully connected layer
    
        def forward(self, X):
            # Forward propagation logic
   ```
    
          
#### Model Training and Evaluation
Gradient clipping is a method of restricting gradients to prevent the occurrence of **gradient explosion** and to avoid affecting model training.
The specific cutting method is shown in the following formula:

$$g \leftarrow min(1, \frac{\theta}{||g||})g$$

Where,  $||g||$ represents **the two norm of gradient**, and  $\theta$ represents **setting range**.
   ```python
    # Gradient clipping (prevent gradient explosion)
    def grad_clipping(net, theta)
    
    # Model evaluation
    def evaluate_net(net, train_iter, test_iter, device)
    
    # Model training
    def train(net, train_iter, test_iter, loss, updater, num_epochs, device)
   ```
Some running results of the final model are shown in the figure
![输入图片说明](/img/图片6.png)

Use the classification_report() function to get the pre
Measure the index values of **accuracy**, **precision**, **recall** and **F1 score**.
![输入图片说明](/img/图片1.png)

Confusion matrix

![输入图片说明](/img/图片2.png)

It is easy to observe that the overall evaluation accuracy is exactly equal to the proportion of Category 3 in the total test set. Additionally, the first and second columns of the confusion matrix are entirely zero, indicating that the model did not classify any corpus into Category 1 or Category 2. This makes it clear that **the model lacks the ability to recognize Categories 1 and 2**, and it simply predicts all data as Category 3.

This issue arises because Category 3 accounts for a disproportionately large share of the total sample pool. To address this, we need to adjust the data volume in both the training and test sets. The goal of this adjustment is to ensure that the model achieves **non-zero prediction accuracy for all three categories**—meaning it gains the ability to predict each category effectively. After such adjustments, the model’s actual accuracy stands at approximately 70%.
![输入图片说明](/img/图片3.png)
