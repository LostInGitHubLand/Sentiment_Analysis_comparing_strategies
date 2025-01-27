# Sentiment_Analysis_comparing_strategies



## Aim of this work
A comparison of different strategies for sentiment analysis is presented.
The following models are selected for comparison:
-	Machine learning (ML) traditional algorithms (Logistic regression, Random Forest, Stochastic Gradient Descent and Bernoulli Naive Bayes);
-	Deep learning networks (CNN);
-	Transformers (BERT (base) and XLNET (base)).

In order to ascertain the most efficacious embedding/vectorization technique, a series of models were subjected to rigorous testing.
Vectorisation:
- Count Vectorisation
- TF-IDF
Embedding:
- Word2vec: trained on dataset and a pretrained version on Google-300-news

## Benchmark
The dataset employed for the purpose of establishing a benchmark is as follows: [dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)
The objective of the preliminary data processing stage was to generate a training, testing and validation dataframe. 
It is important to note that all of the datasets were **balanced**, and the sentiment analysis involved **three** categories: negative, neutral and positive.

## Sentiment Analysis

![image](https://github.com/user-attachments/assets/2ba53663-f084-4b08-b18c-f512381cbcb5)



## Evaluation metric and results
The **accuracy** score is utilised as the metric for evaluation.
Comparison of different embedding/vectorization techniques with traditional ML models.
| | Accuracy | (optimum) ML algorithm | 
|-|-|-| 
| Embedding (w2vec-training data)|  48%    | SGD | 
| Embedding (w2vec pre-trained)|    64%    | SGD |||
| **CountVectorizer** |                **70.4%**   | **SGD** |
| TF-IDF Vectorizer|               70.3%    | SGD |


| | Accuracy | 
|-|-|
| SGD| 70.4% |
|CNN| |
| BERT| 78.4%|
|XLNET | 79.6%|      



# Bibliography
[](file:///home/profpao/Scaricati/SA_article.pdf)
