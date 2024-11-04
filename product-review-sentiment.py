# General packages
import numpy as np
import pandas as pd

# NLP packages
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Modeling packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#Text normilization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('punkt_tab')
from nltk.stem import PorterStemmer, LancasterStemmer # Common stemmers
from nltk.stem import WordNetLemmatizer # Common Lematizer

product_reviews = pd.read_csv('C://bjai/gitrepo/bj-ml/Customer-Product-Review.csv')
##pre-processing
#convert word into lowercase
product_reviews['reviews_text']=product_reviews['review_body'].str.lower()
#remove special char
#review_backup = product_reviews['reviews_text_new'].copy()
product_reviews['reviews_text'] = product_reviews['reviews_text'].str.replace(r'[^A-Za-z0-9 ]+', ' ')
#remove stop words and stemm or lemmatize
def tokenize_stemm_terms(text):
    stop_words_1 = set(stopwords.words('english'))
    words = word_tokenize(text)
    words_stemm = [LancasterStemmer().stem(w) for w in words if w not in stop_words_1]
    return words_stemm

##Vectoriztaion
tfidf_vectorizer = TfidfVectorizer(tokenizer= tokenize_stemm_terms, # type of tokenization
                             ngram_range=(1,1), token_pattern=None) # number of n-grams)
v_data = tfidf_vectorizer.fit_transform(product_reviews['reviews_text'].values.astype('U'))
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(v_data, # Features
                                                                    product_reviews['sentiment'], # Target variable
                                                                    test_size = 0.2, # 20% test size
                                                                    random_state = 0) # random state for replication purposes
##Apply logistic regression
#Training the model 
lr_model = LogisticRegression() # Logistic regression
lr_model.fit(X_train_data, y_train_data) # Fitting a logistic regression model

##Predicting the output
test_predict = lr_model.predict(X_test_data) # Class prediction

##Calculate key performance metrics
print("F1 score: ", f1_score(y_test_data, test_predict))