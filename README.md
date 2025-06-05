# Sentiment_Analysis_comparing_strategies

# Recomandation
Download from this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g) the GoogleNews vectors negative 300 binary file.
It is possible to install all the necessary dependencies for a given virtual environment using the **requirements.txt** file.

*Example*. Setting the enviroment using **virtualenv** (linux user).

`virtualenv -p /usr/bin/python3.12 sa_env`

Then activate the env

`sa_env activate`

Finally install packages

`pip install -r requirements.txt`

## Aim of this work
A comparison of different strategies for sentiment analysis is presented.

The following models are selected for comparison:
-	Machine learning (ML) traditional algorithms (Logistic regression, Random Forest, Stochastic Gradient Descent and Bernoulli Naive Bayes);
-	Deep learning networks (Naive architecture, CNN);
-	Transformers (**BERT** (base) and **XLNET** (base)).

In order to ascertain the most efficacious embedding/vectorization technique, a series of models were subjected to testing.
- Vectorisation:
    - Count Vectorisation
    - TF-IDF
- Embedding:
    - Word2vec:
      - trained on the dataset
      - pretrained version on Google-300-news

## Dataset
The dataset employed for the purpose of establishing a benchmark is as follows: [dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)
The objective of the preliminary data processing stage was to generate a cleaned training, testing and validation dataframe. 
It is important to note that all of the datasets were **balanced**, and the sentiment analysis involved **three** categories: negative, neutral and positive.
![image](https://github.com/user-attachments/assets/0e2f86fa-1d4c-4408-95d8-24b5c95d4d8a)


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



