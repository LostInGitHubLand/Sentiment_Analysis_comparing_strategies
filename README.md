# Sentiment_Analysis_comparing_strategies



## Aim of this work
A comparison of different strategies for sentiment analysis is presented.
The following models are selected for comparison:
-	Machine learning (ML) traditional algorithms (Logistic regression, Random Forest, Stochastic Gradient Descent and Bernoulli Naive Bayes);
-	Deep learning networks (Naive architecture, CNN);
-	Transformers (**BERT** (base) and **XLNET** (base)).

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
Comparison of different embedding/vectorization techniques with **traditional ML models**.
In the present study, the following machine learning algorithms were tested.
- Logistic regression,
- Random Forest,
- Stochastic Gradient Descent (SGD)
- Bernoulli Naive Bayes
The following section will present the results obtained from the different vectorization/embedding implementations of the optimal algorithm: SGD.

| | Accuracy | (optimum) ML algorithm | 
|-|-|-| 
| Embedding (w2vec-training data) |  48%    | SGD | 
| Embedding (w2vec pre-trained) |    64%    | SGD |
| **CountVectorizer** |   **70.4%**   | **SGD** |
| TF-IDF Vectorizer|               70.3%    | SGD |


In the following section, the concluding results of the study will be presented.
| | Vectorization/embedding| Accuracy | 
|-|-|-|
| SGD| Vectorization|70.4% |
| Naive Deep Learning (15epochs)| Embedding (sentence transformer)| 70.3%|
|CNN (5epochs) |Embedding (sentence transformer)| 69.3%|
| BERT| Embedding(token embedding + sentence embedding + positional encoding)| 78.4%|
|**XLNET** |Embedding (token embedding + sentence embedding + positional encoding) | **79.6%**|      



