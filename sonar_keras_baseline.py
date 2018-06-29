#Binary classification using Sonar Dataset : baseline

import numpy as np
import pandas as pd
from  keras.models import  Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)

#load dataset
dataframe=pd.read_csv("sonar.csv" , header=None)
dataset = dataframe.values

#split the input variables(X) and the output variables(Y)
X = dataset[:,:-1].astype(float)
y=dataset[:,-1]

#encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

#baseline model
#[input 60]-->[hidden 60] --> [output 1 (binary) : either Rock or Mines]
def create_baseline():
	#create model
	model = Sequential()
	model.add(Dense(60 , input_dim= 60 , kernel_initializer= 'normal', activation = 'relu'))
	model.add(Dense(1 , kernel_initializer = 'normal', activation ='sigmoid'))
	#compile model
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model
# evaluate model

estimator = KerasClassifier(build_fn= create_baseline, epochs =100 , batch_size = 5 , verbose =0)
kfold = StratifiedKFold(n_splits=10 , shuffle = True ,random_state=42)
results = cross_val_score(estimator , X , encoded_Y , cv = kfold)
print("Baseline" , results.mean() , results.std())


