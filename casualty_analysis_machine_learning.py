# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 14:58:05 2021

@author: alanm
"""

project_path = "C:/Users/alanm/OneDrive - National College of Ireland/Final Year/Software Project/Project"

# libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np



plt.style.use("classic")
os.chdir(project_path)

""" ------- FILE TO RUN THROUGH ML PIPELINE """

in_file = "ml_set"

with open(f"./data/pickles/{in_file}.pkl", "rb") as f:
    ml_set = pickle.load(f)


""" -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-"""


df = ml_set.copy()

for c in df.columns:
    print(c, df[c].nunique())
    
for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())

# drop accident severity


# Create Train and Test Sets -------------------------------------------------

target_var = "casualty_severity"
dropping_var = "accident_severity"


# target_var = "accident_severity"
# dropping_var = "casualty_severity"



df = ml_set.drop(dropping_var, axis=1)

""" QUICK FIX FOR NEGATIVE LONGLATS  """
df["adj_long"] = df["adj_long"]*df["adj_long"]
df["adj_lat"] = df["adj_lat"]*df["adj_lat"]




# stratified split


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 1)

for train_index, test_index in split.split(df, df[target_var]):
    train = df.loc[train_index]
    test = df.loc[test_index]


# Column Transformer for category columns

## Separate Labels



# predictors (X)
X_train_set = train.drop(target_var, axis=1)
X_test_set = test.drop(target_var, axis=1)

# labels (y)
y_train = train[target_var].copy()
y_test = test[target_var].copy()



# instantiate one hot encoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()


# columns for encoding

cat_cols = ["number_of_vehicles_bin", "number_of_casualties_bin", "lad_code", "engine_capacity_group", "age_band_of_vehicle"]




from sklearn.compose import ColumnTransformer
cat_transf = ColumnTransformer([
    ("cat", OneHotEncoder(), cat_cols),],
    remainder = "passthrough")


X_train = cat_transf.fit_transform(X_train_set)
X_test = cat_transf.fit_transform(X_test_set)



# X_train = cat_transf.fit_transform(X_train_set)
# X_test = cat_transf.fit_transform(X_test_set)



# # Feature Selectin Review -----------------------------------------------

# # should do this at an earlier stage, but testing here on the prepared set

# from sklearn.feature_selection import SelectKBest, chi2
# fs = SelectKBest(score_func=chi2, k='all')
# fs.fit(X_train, y_train)
# X_train_fs = fs.transform(X_train)
# X_test_fs = fs.transform(X_test)

# for i in range(len(fs.scores_)):
# 	print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
# plt.show()


# mask = fs.get_support() #list of booleans
# new_features = [] # The list of your K best features

# feature_names = list(df.columns.values)

# for bool, feature in zip(mask, feature_names):
#     if bool:
#         new_features.append(feature)
 
# column_names = d.columns[chY.get_support()]
 
# Train Models ----------------------------------------------------------


# """ GRADIENT BOOSTING MACHINE """
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix

# gbc_classifier = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,
#     max_depth=100, random_state=1)

# gbc_classifier.fit(X_train, y_train)

# gbc_classifier.score(X_test, y_test)

# # get score for model
# gbc_predictions = gbc_classifier.predict(X_test)
# print("Gradient Boosting Machine:", metrics.accuracy_score(y_test, gbc_predictions))
# confusion_matrix(y_test, gbc_predictions)




# # pickling processed data
with open (f"./data/pickles/casualty_gbc.pkl", "wb") as f:
    pickle.dump(gbc_classifier, f)


""" DECISION TREE """
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# get score for model
dt_predictions = dt_classifier.predict(X_test)
print("Decision Tree Accuracy:",metrics.accuracy_score(y_test, dt_predictions))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
confusion_matrix(y_test, dt_predictions)



""" RANDOM FOREST """
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state = 1)
rf_classifier.fit(X_train, y_train)

# get score for model
rf_predictions = rf_classifier.predict(X_test)
rf_score = rf_classifier.score(X_test, y_test) * 100

print(rf_score)

print("Random Forest Accuracy:",metrics.accuracy_score(y_test, rf_predictions))
confusion_matrix(y_test, rf_predictions)


""" KERNEL DENSITY """
from sklearn.neighbors import KernelDensity


kde_classifier = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde_classifier.fit(X_train, y_train)

# get score for model
kde_score = kde_classifier.score_samples(X_test)


""" GAUSSIAN NAIVE BAYES """
from sklearn.naive_bayes import GaussianNB

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

# get score for model
gnb_predictions = gnb_classifier.predict(X_test)
gnb_score = gnb_classifier.score(X_test, y_test) * 100

print("Gaussian Naive Bayes Accuracy", metrics.accuracy_score(y_test, gnb_predictions))
confusion_matrix(y_test, gnb_predictions)


# """ K NEAREST NEIGHBOUR """
# from sklearn.neighbors import KNeighborsClassifier


# knn_classifier = KNeighborsClassifier(n_neighbors = 1)
# knn_classifier.fit(X_train, y_train)

# # get score for model
# knn_predictions = knn_classifier.predict(X_test)
# knn_score = knn_classifier.score(X_test, y_test) * 100

# print(knn_score)

# # confusion matrix
# print("K Nearest Neighbour Accuracy:",metrics.accuracy_score(y_test, knn_predictions))
# confusion_matrix(y_test, knn_predictions)


# """ LINEAR DISCRIMINANT ANALYSIS """
# from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis


# lda_classifier = LinearDiscriminantAnalysis()
# lda_classifier.fit(X_train, y_train)

# # get score for model
# lda_predictions = lda_classifier.predict(X_test)
# lda_score = lda_classifier.score(X_test, y_test) * 100

# print("Linear Discriminant Analysis Score:", lda_score)


""" NEURAL NETWORK MLP Classifier """
from sklearn.neural_network  import MLPClassifier


mlp_classifier = MLPClassifier(random_state=1, max_iter=150)
mlp_classifier.fit(X_train, y_train)

# get score for model
mlp_predictions = mlp_classifier.predict(X_test)
mlp_score = mlp_classifier.score(X_test, y_test) * 100

# confusion matrix
print("Neural Network Accuracy:",metrics.accuracy_score(y_test, mlp_predictions))
confusion_matrix(y_test, mlp_predictions)



# Keras fully connected Network





from keras.models import Sequential
from keras.layers import Dense



# create model
# model = Sequential()
# model.add(Dense(512, activation="tanh", kernel_initializer="uniform"))
# model.add(Dense(512, activation="linear", kernel_initializer="uniform"))
# model.add(Dense(1))


seq_classifier = Sequential()
seq_classifier.add(Dense(512))
seq_classifier.add(Dense(512))
seq_classifier.add(Dense(1))


# Compile model
seq_classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
seq_classifier.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=5, batch_size=10,  verbose=2)

# get score for model
seq_predictions = seq_classifier.predict(X_test)
seq_score = metrics.accuracy_score(X_test, y_test) * 100

# confusion matrix
print("Keras Sequential Classifier:",metrics.accuracy_score(y_test, seq_predictions))
confusion_matrix(y_test, mlp_predictions)

""" I DONT UNDERSTAND WTF IS GOING ON HERE """





