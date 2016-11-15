# Example output is also included for each question.
# Note that there is randomness involved (both in how the
# data is split and also in the Random Forest), so you will not always get
# exactly the same results.


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.feature_extraction.text import TfidfVectorizer



# 1. Load the dataset in with pandas
def helper(p):
    data = []
    path = p
    files = [f for f in os.listdir(path) if os.path.isfile(path+f)]
    for f in files:
        with open (path+f, "r") as myfile:
            data.append(myfile.read())
    return pd.DataFrame(data)

df_pos = helper("/Users/naveenanair/Downloads/review_polarity/txt_sentoken/pos/")
df_neg = helper("/Users/naveenanair/Downloads/review_polarity/txt_sentoken/neg/")
df_neg["comments" ]= df_neg[0]
df_neg = df_neg.drop(0,axis = 1)
df_pos["comments" ]= df_pos[0]
df_pos = df_pos.drop(0,axis = 1)

df_app = df_neg.append(df_pos)


# 2. Make a target value y called PositiveSentiment containing the 0/1 for neg reviw/pos review
df_neg["PositiveSentiment"] = 0
df_pos["PositiveSentiment"] = 1

# 3. Make a 2 dimensional numpy array containing the feature data (everything except the labels)
df_app = df_neg.append(df_pos)


# 4. Clean the dataframe containing the comments to remove ascii charaters
df_app['comments'] = df_app['comments'].apply(lambda x: x.decode('unicode_escape').\
                                              encode('ascii', 'ignore').\
                                              strip())

# 5. Use sklearn's train_test_split to split into train and test set
X_train, X_test, y_train, y_test = train_test_split(df_app['comments'], df_app['PositiveSentiment'], random_state=1)


# 6. Perform word to vec conversion
tfidf = TfidfVectorizer(stop_words='english')
XT_train = tfidf.fit_transform(X_train)
XT_test = tfidf.transform(X_test)


# 7. Use sklearn's RandomForestClassifier to build a model of your data
rf = RandomForestClassifier()
rf.fit(XT_train, y_train)


# 8. What is the accuracy score on the test data?
print "8. RandomForestClassifier model score:", rf.score(XT_test, y_test)
# ## answer: 0.65

# 9. Draw a confusion matrix for the results
y_predict = rf.predict(XT_test)
print "9. RandomForestClassifier model confusion matrix:"
print confusion_matrix(y_test, y_predict)
# ## answer:  202   53
# ##           108  137
#
# 10. What is the precision? Recall?
print "10. RandomForestClassifier model precision:", precision_score(y_test, y_predict)
print "  RandomForestClassifier model recall:", recall_score(y_test, y_predict)
## precision:  0.761627906977
##    recall: 0.534693877551
#
# 11. Build the RandomForestClassifier again setting the out of bag parameter to be true
rf = RandomForestClassifier(n_estimators=30, oob_score=True)
rf.fit(XT_train, y_train)
print "11: Modified RandomForestClassifier model accuracy score:", rf.score(XT_test, y_test)
print "    Modified RandomForestClassifier model out of bag score:", rf.oob_score_
##   accuracy score: 0.768
## out of bag score: 0.684666666667   (out-of-bag error is slightly worse)

# 12. Try modifying the number of trees
num_trees = range(5, 500, 5)
accuracies = []
for n in num_trees:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(XT_train, y_train)
        tot += rf.score(XT_test, y_test)
    accuracies.append(tot / 5)
plt.plot(num_trees, accuracies)
plt.show()
#Accuracy increases as the number of trees increase optimal accuracy 500 trees

# 13. Run all other classifiers
def get_scores(classifier, XT_train, XT_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(XT_train, y_train)
    y_predict = model.predict(XT_test)
    return model.score(XT_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

print "16. Model, Accuracy, Precision, Recall"
print "    Random Forest:", get_scores(RandomForestClassifier, XT_train, XT_test, y_train, y_test, n_estimators=25, max_features=5)
print "    Logistic Regression:", get_scores(LogisticRegression, XT_train, XT_test, y_train, y_test)
print "    Decision Tree:", get_scores(DecisionTreeClassifier, XT_train, XT_test, y_train, y_test)
print "    SVM:", get_scores(SVC, XT_train, XT_test, y_train, y_test)
print "    Naive Bayes:", get_scores(MultinomialNB, XT_train, XT_test, y_train, y_test)
## MODEL               ACCURACY PRECISION    RECALL
## Random Forest         0.68    0.66   0.70
## Logistic Regression   0.80    0.78    0.82 - LogisticRegression model performs best
## Decision Tree         0.63    0.62    0.62
## SVM                   0.48    0.48    0.48
## Naive Bayes           0.78    0.79    0.75 - Naive Bayes model performs second best
