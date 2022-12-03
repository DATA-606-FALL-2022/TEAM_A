# TEAM_A
# PROPAGANDA NEWS CLASSIFICATION USING ML & DEEP LEARNING TECHNIQUES
![Alt text](https://github.com/DATA-606-FALL-2022/TEAM_A/blob/main/images/Screenshot_20221029_022630.png)
This project is dedicated to doing some cool data analysis, visualization,eda and modeling using ML and Deep learning models.
## Repo Contents
  <li><b>Data:</b>This folder  contains  data sets used in our project and origin of data
  <li> https://drive.google.com/drive/folders/1wQHY3DJwhhGTK4lHtWmVks33P4X-p-tb?usp=sharing</li>
   <li> https://zenodo.org/record/3271522#.Yxg-FuxN2SV.</li>
 
 ### Link to the PPT:
 https://docs.google.com/presentation/d/1K-DRitsWeXgu9it_Uy2zd_ajrpe3IBVIpLaZRgy--5M/edit#slide=id.g17d5c09ff22_1_38

### You tubeVideo link:
https://youtu.be/wyj-9mT6p8A</li>
 
 ### TEAM MEMBERS
<li><b>Sonia Sonia</b></li>
<li><b>Seshadivya Batlanki</b></li>

 ### PROJECT OVERVIEW :
<li>ABSTRACT 
<li>INTRODUCTION
<li>LITERATURE REVIEW
<li>PROPOSED METHODOLOGY
<li>DATA SET
<li>EDA
<li>MODELS
<li>MACHINE LEARNING MODELS USED
<li>FUTURE WORK
<li>REFERENCES
  
  ### ABSTRACT:
  By offering a platform for the distribution of knowledge, social networking has taken over the entire planet. Most of the time, people pass along information without verifying its accuracy. Social media platforms are now utilized to influence decisions in a variety of areas, including politics, advertising, and more. It is hardly unexpected that misinformation is being circulated on social media as a means of swaying public opinion. One methodical and intentional strategy used to persuade people for political or religious ends is propaganda. In this study, machine learning and deep learning algorithms were used to attempt to distinguish between propaganda and non-propaganda text.
### INTRODUCTION:
  Computer science is crucial in today's technological age for offering answers to practically all new sectors. Computer science has advanced dramatically since the 1970s, when the internet first appeared. It is now used in a variety of multidisciplinary fields, including remote sensing, technical diagnosis, traffic control systems, criminology, medical imaging, image processing, data mining, and automatic surveillance. The market is seeing a huge increase in the number of hardware and software products because of these applications. Today, data analytics is a key area of study for finding patterns in massive data sets. It integrates with several disciplines, including bioinformatics, natural language processing (NLP), machine learning, and others. Important information is extracted from text, images, videos, and other sources during data mining. Data mining can perform both descriptive and predictive tasks. Descriptive tasks are used to characterize data, while predictive jobs use historical data to estimate the future. Some of the tasks involved in data mining include clustering, correlation, and pattern finding. Online social network analysis is a difficult procedure because of the massive utilization, variety, volume, validity, and real-time data that these networks generate. Online social networks (OSN) communicate using computer-mediated tools that facilitate the creation and dissemination of knowledge, ideas, business benefits, and novel communication techniques through online communities and links.
### Research Question:
  <li>Why it is required to detect propaganda based news?</li>
  <li>Which deep learning model can be used for efficient classification of propaganda based news?</li>
  <li>How to create web application that can handle large dataset with proper data structure format?</li>
  <li>Does contextual and linguistic features matter for such news classification?</li>

### DATASET:
The dataset comprises data extracted from https://zenodo.org/record/3271522#.Yxg-FuxN2SV .This project aims to build an efficient web based prototype for users which can classify the propaganda based news and articles from the actual ones.

![Alt text](https://github.com/DATA-606-FALL-2022/TEAM_A/blob/main/images/download.png)
![Alt text](https://github.com/DATA-606-FALL-2022/TEAM_A/blob/main/images/download%20(1).png)

### Data Cleaning and Text-Pre-Processing
 Below are the steps being performed as a part of text preprocessing.
* 1.Lowercasing each row
* 2.Deconstruction of English words (ex: can’t – cannot)
* 3.Removal of special characters and punctuations
* 4.Removal of numbers and digits
* 5.Removal of stop words
* 6.Lemmatization of each word in each row

### Word cloud analysis

A word cloud, also known as a tag cloud, is a graphic representation of text data in the form of tags. These tags are usually single words, and the size and color of the words indicate how important they are. The need to analyze the enormous amounts of text produced by these systems is growing as unstructured data in the form of text continues to see unprecedented development, particularly within the field of social media. By showing the word frequency in the text as a weighted list, a word cloud is a fantastic tool for aiding in the visual interpretation of literature and is helpful in swiftly getting insight into the most important elements in a particular text.

![Alt text](https://github.com/DATA-606-FALL-2022/TEAM_A/blob/main/images/download%20(3).png)
![Alt text](https://github.com/DATA-606-FALL-2022/TEAM_A/blob/main/images/download%20(2).png)

### Primary Unit of Analysis:
Propaganda based news detection and classification is one of the challenging task.
<li>(1) Because it is written for a particular purpose and particular group of people and</li>
<li>(2)Because of similar use of context and linguistic features written for normal news as well. To solve these problems, this project aims to build an efficient web application where user can do a check whether the article, news is based on propaganda or not.</li>


### FEATURE SELECTION:
* Functional Requirements
* User should upload the data.
* User should be able to see the classified results.
* The proper user interface should be provided in web based prototype.
* Non - Functional Requirements
 * Deep learning model for detection should be accurate.
 * The prototype should be user friendly.

### METHODOLOGY:
* Selection of high-quality input data.
* Pre-processing, and creation of clean data.
* Implementation of contextual and other embeddings.
* Implementation of neural models using LSTM, attention, BERT, etc. models
* Validation of model using log loss and f1, precision, recall, auc_roc scores
* End to end pipeline creation for prediction of translated language at real time.

### MODEL SELECTION:
* Prediction will be done considering the best performed model.
* Building a prototype for the predictive model using deep learning or machine learning which can classify the propaganda news.

### OUTCOME:
* The deep learning model should be efficient and accurate:

### TEAM RESPONSIBILITIES

#### Sonia Sonia:
* Data collection and dataset extraction.
* Initial Cleaning of the data.
* Exploratory Data Analysis.
* Performing Regression analysis.

#### Seshadivya Batlanki:
 * Data Modeling
* Simulating different prediction models.
* Model comparison and Cross validating the results.
* Finding a best fit model for prediction.

### Unit of Analysis:
Our unit of analysis is to classify the propaganda and non-propaganda articles and news present in the dataset using NLP techniques.
The type of variable present in the dataset is news, articles-based text. We are going to use various NLP techniques to handle text data, such as pre-processing using removal of stop words, stemming, etc., pre-processing based on pre-trained embeddings such as fasttext, glove, word2vec, vectorization using tfidf, count and word encoding.
There will be various machine learning models going to be built on the vectorized dataset. Models are:
1.	Naïve Bayes
2.	Logistic Regression
3.	Decision Tree
4.	Random Forest
5.	AdaBoost
6.	XGBoost
We are also going to build deep learning-based models and architectures which involves the algorithms below:
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


### Conclusion and Future works:
Given the impact this unregulated content has on society, the significance of spotting propaganda is clear. This study intends to develop a system for identifying propaganda utilizing a convolution LTSM Bases Deep Learning Model network. To anticipate accurately, the neural network trains itself over many iterations. This model's efficiency in testing was 97.58% after processing a dataset of more than 52k+ news articles. To demonstrate that the proposed system is superior to the n-gram models, a comparison study is conducted.
The analysis of propaganda project has a promising future. It can be extended to include categorizing the test piece according to the kinds of propaganda strategies employed. It will improve the model's specificity, broadening its application. In social media and other online platforms where there is a high risk of fake news, propaganda detection can be employed to prevent swaying public opinion for private advantage. To preserve original viewpoints in society and uphold the democratic ideal, news policing will be automated.

