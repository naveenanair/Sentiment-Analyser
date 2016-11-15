
## Sentiment Analysis 
This sentiment analysis model tries to predict whether a certain review is positive or negative. It uses the tf-idf word to vec transformation to convert textual reviews into a matrix representation of themselves. Several models are then analyzed to determine which ones perform best across the three metrics of accuracy, precision and recall. The random forest model is explored a bit more by performing fine tuning of model parameters to optimize model performance. 

## Data Source 
The movie review data set (http://www.cs.cornell.edu/people/pabo/movie-review-data/) was used as the data scource for this project. The data set has classified sets of positive and negative reviews which were used to train and test the models. 

## Toolkit
Tools used to complete this project include Python, numpy, pandas, sklearn, matplotlib

## Data Pipeline 
1. Positive and negative tect reviews were loaded into separate data frames 
2. A column called "PositiveSentiment" was added, which was 0 for negative reviews and 1 for positive reviews, this was the target Y variable  
3. Both data frames were appended to form a consilidated set of positive and negative reviews with their classification as 0/1 (neg/pos)
4. The data set was split into a training and test data set in order to cross validate models and evaluate model performance 

## Data Models 
1. For performing word to vec transformation two vectorizers were evaluated from the sklearn toolkit CountVectorizer and TfidfVectorizer. For the same model data vectorized using the TfidfVectorizer gave a better accracy score therefore, it was used to vectorize all data for the models. 
2. Models tested include 
- RandomForestClassifier 
- LogisticRegression
- SVM
- DecisionTrees
- NaiveBayes Classifier 
3. The models were tested on three metrics , accuracy, precision and recall. For all three metrics the LogisticRegression model performed best followed by the Naive Bayes Model. Sample results are included below 

MODEL               ACCURACY PRECISION    RECALL
Random Forest         0.68    0.66   0.70
Logistic Regression   0.80    0.78    0.82 - LogisticRegression model performs best
Decision Tree         0.63    0.62    0.62
SVM                   0.48    0.48    0.48
Naive Bayes           0.78    0.79    0.75 - Naive Bayes model performs second best
