#import libraries

import numpy as np
import pandas as pd


#importing dataset

dataset = pd.read_csv('/Users/mohsen/Desktop/data-preprocessing/Data-preprocessing.csv')
x = dataset.iloc[:,:3].values
y = dataset.iloc[:,-1].values

print(x)
print(y)

#taking care of missing data

from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
Imputer.fit(x[:,1:3])
x[:,1:3] = Imputer.transform(x[:,1:3])

print(x)

#encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('Encoder',OneHotEncoder(), [0])] , remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

print(x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

#spliting the dataset to training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)


