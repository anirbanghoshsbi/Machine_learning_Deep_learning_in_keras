#multi class flower classification

#import the necessary packages
import numpy as np
from csv import reader
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
import argparse

# load the data



def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


#define a baseline model
def baseline_model():
	# [input 4]-->[hidden 8]---> [hidden 8] ---> [3 output  : categorical]
	model = Sequential()
	model.add(Dense(8 , input_dim=4, activation ='relu'))
	model.add(Dense(8 , activation ='relu'))
	model.add(Dense(3 , activation ='softmax'))
	model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#main program :
seed= 7
ap=argparse.ArgumentParser()
ap.add_argument('-d' , '--dataset', required =True , help= "path to the file")
args=vars(ap.parse_args())


#load the data
filename = args['dataset']
dataset = load_csv(filename)

# converting string to floats
for i in range(4):
	str_column_to_float(dataset, i)
# convert class column to int
lookup = str_column_to_int(dataset, 4)

# convert the dataset to numpy array	
dataset=np.array(dataset)
X= dataset[:,:-1]
y=dataset[:,-1]

#  KerasClassifier class internally passes on to the fit function with the necessary parameters and once the
# function is fit we use the cross_val_score for estimating the mean and standard deviation .
estimator= KerasClassifier(build_fn = baseline_model , epochs = 100 , batch_size=15, verbose =2)
kfold=KFold(n_splits =10 , shuffle = True, random_state= seed)
results= cross_val_score(estimator , X, y ,cv=kfold)
print("Accuracy mean {} , std dev{}:".format(results.mean()*100 , results.std()*100)) 
	

		
