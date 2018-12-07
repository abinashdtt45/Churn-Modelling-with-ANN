

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X= dataset.iloc[:, 3:13].values
Y= dataset.iloc[:, 13].values

X_cat = dataset['Geography']
X_in = pd.get_dummies(X_cat, drop_first=True)
X_cats = pd.DataFrame(X_in)
#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
x2 = LabelEncoder()
X[:, 2] = x2.fit_transform(X[:, 2])
X.shape
X=X[:,[0, 2, 3, 4, 5, 6, 7, 8, 9]]
X_news = pd.DataFrame(X)
X_new = pd.concat([X_cats,X_news], axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Buildin the ann
import keras
from keras.models import Sequential
from keras.layers import Dense
#normal model with out any optimisation
classifier= Sequential()

classifier.add(Dense(6, activation="relu", input_dim=11, kernel_initializer="uniform"))
classifier.add(Dense(6, activation="relu", kernel_initializer="uniform"))
classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, Y_train, batch_size=10, epochs=100)


y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)


#k-fold validation to check whether the acuracy was the average or lucky

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation="relu", input_dim=11, kernel_initializer="uniform"))
    classifier.add(Dense(6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
mean = accuracies.mean()
variance = accuracies.std()

#do drop out if overfitted that is variance is high
#Improving the paramas
#Using gridsearch to find the best parameter that can bring the best result.
parameters = {'batch_size' : [10, 25, 16, 32],
              'epochs' : [100, 200, 500], 
              'optimizer' : ['adam', 'rmsprop']}

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation="relu", input_dim=11, kernel_initializer="uniform"))
    classifier.add(Dense(6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv = 10)
grid_search=grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

