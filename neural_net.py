import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('../Basic_Nueral_Network_model-Titanic_Dataset/train.csv')
test_data = pd.read_csv('../Basic_Nueral_Network_model-Titanic_Dataset/test.csv')

print(train_data.shape, test_data.shape)

train_data=train_data.fillna(train_data.mean())
test_data=test_data.fillna(test_data.mean())

label_encoder = LabelEncoder()
train_data.iloc[:,4] = label_encoder.fit_transform(train_data.iloc[:,4])
test_data.iloc[:,3] = label_encoder.fit_transform(test_data.iloc[:,3])

train_data = train_data.drop([ 'Embarked', 'PassengerId','Name', 'Cabin', 'Ticket'], axis=1)
test_data = test_data.drop([ 'Embarked', 'PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)


f_array = train_data[['Pclass', 'Sex', 'Age','Fare','SibSp', 'Parch']].values
target_value = train_data['Survived'].values
fe_array = test_data[['Pclass', 'Sex', 'Age','Fare', 'SibSp', 'Parch']].values

X_train, X_test, y_train, y_test = train_test_split(f_array, target_value, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100)

accuracy = model.evaluate(X_train, y_train)
print(accuracy)

predict_value = model.predict_classes(X_test)
accu = accuracy_score(y_test, predict_value.round())
print(accu)

for i in range(5):
	print(' %d (expected %d)' % (predict_value[i], target_value[i]))


predict_value_test = model.predict_classes(fe_array)
accu2 = accuracy_score(target_value[:418], predict_value_test.round())
print(accu2)

for i in range(5):
	print(' %d (expected %d)' % ( predict_value_test[i], target_value[i]))

