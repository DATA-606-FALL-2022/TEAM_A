# TEAM_A
# DATA 606 PROJECT PROPOSAL

# PROPAGANDA NEWS CLASSIFICATION USING DEEP LEARNING TECHNIQUES

## TEAM DESCRIPTION

This is a team project of Sonia Sonia and Seshadivya Batlanki.


## PROJECT OVERVIEW

* Propaganda spreads the ideology and beliefs of like-minded people, brainwashing their audiences, and sometimes leading to violence. 
* Due to common reach of internet to the people, it has become easy for many sources to manipulate people to believe in something which is toxic and unhealthy by the use of the propaganda.
* Propaganda news looks like the original news and it’s very difficult to identify as it is written very carefully. There are different ways through which we can identify it. Such as-
•	Having good knowledge over the topic
•	Focusing on words and topics where the stress is more.


## BACKGROUND

Though propaganda news classification is not a recent topic, but is an important one and in research for a long time. As the increase in social media use, there are many platforms and sources trying to populate their propaganda and feed to the people. The style of writing is also improving. So, there is a high scope of use of state-of-the-art model to identify the news and save many human minds to become toxic. There is a huge scope of improvement in this topic as there is very limited number of models and algorithms being implemented.


## RESEARCH QUESTIONS:


1.  Why it is required to detect propaganda based news?

2.	Which deep learning model can be used for efficient classification of propaganda based news?

3.	How to create web application that can handle large dataset with proper data structure format?

4.	Does contextual and linguistic features matter for such news classification?



## DATASET 

For the analysis, dataset is taken from the link-https://zenodo.org/record/3271522#.Yxg-FuxN2SV.  The dataset is a publicly available text dataset. The corpus contains 52k articles from 100+ news outlets. Each article is labelled as either “propagandistic” (positive class) or “non-propagandistic” (negative class). The labelling was done indirectly using a technique known as distant supervision, i.e., an article is considered propagandistic if it comes
from a news outlet that has been labelled as propagandistic by human annotators.


## PRIMARY UNIT OF ANALYSIS

Propaganda based news detection and classification is one of the challenging task. 

(1) Because it is written for a particular purpose and particular group of people and 

(2)Because of similar use of context and linguistic features written for normal news as well. To solve these problems, this project aims to build an efficient web application where user can do a check whether the article, news is based on propaganda or not.


Unit of Analysis:

Our unit of analysis is to classify the propaganda and non-propaganda articles and news present in the dataset using NLP techniques. 

The type of variable present in the dataset is news, articles based text. We are going to use various NLP techniques to handle text data, such as pre-processing using removal of stop words, stemming, etc. , pre-processing based on pre-trained embeddings such as fasttext, glove, word2vec, vectorization using tfidf, count and word encoding. 

There will be various machine learning models going to be built on the vectorized dataset.
Models are:

1.	KNN
2.	Naïve Bayes
3.	Logistic Regression
4.	SVM 
5.	Decision Tree
6.	Random Forest
7.	XGBoost

We are also going to build deep learning based models and architectures which involves below algorithms.

1.	Recurrent Neural Network
2.	Convolutional neural Network
3.	Attention Models
4.	BERT

As the problem statement is a classification problem below are the evaluation metrics will be used.

1.	Classification Report 
2.	Confusion Matrix
3.	Roc Curve
4.	AUC Score
5.	Precision
6.	Recall
7.	F1 Scores


## FEATURE SELECTION:

1. Functional Requirements

* User should upload the data.
* User should be able to see the classified results.
* The proper user interface should be provided in web based prototype.

2. Non - Functional Requirements

* Deep learning model for detection should be accurate.
* The prototype should be user friendly.


 ## METHODOLOGY:

* Selection of high-quality input data.
* Pre-processing, and creation of clean data.
* Implementation of contextual and other embeddings.
* Implementation of neural models using LSTM, attention, BERT, etc. models
* Validation of model using log loss and f1, precision, recall, auc_roc scores
* End to end pipeline creation for prediction of translated language at real time.

## MODEL SELECTION: 

* Prediction will be done considering the best performed model.
* Building a prototype for the predictive model using deep learning or machine learning which can classify the propaganda news.



## OUTCOMES


* We should properly classify propaganda news from the trained domain of data.
* The deep learning model should be efficient and accurate

## TEAM RESPONSIBILITIES

### Sonia Sonia: 

* Data collection and dataset extraction.
* Initial Cleaning of the data.
* Exploratory Data Analysis.
* Performing Regression analysis.

### Seshadivya Batlanki:

* Data Modeling
* Simulating different prediction models.
* Model comparison and Cross validating the results.
* Finding a best fit model for prediction.


### Benefits:
Along with our respective duties, we shall support one another's efforts to complete the project successfully. we will learn about collaboration, which would be handy for real time work. To advance the project and get around obstacles, we hold frequent discussions.
With this project, we hope to gain knowledge on how to use a machine learning, a project from start to finish while applying feature engineering, working on various categorization models, and evaluating them.

