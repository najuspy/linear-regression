'''
Library Imports
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#Library to Make a Normally Distributed Regression Dataset
from sklearn.datasets import make_regression
#Library to split our dataset into train and test model
from sklearn.model_selection import train_test_split
#Importing libraries for metrics 
from sklearn.metrics import r2_score, mean_squared_error
#Importing our custom made Linear Regression Class
from linear_regression import LinearRegression

'''
Creating our dataset and splitting the dataset
'''
X,y = make_regression(n_samples = 300 , n_features=1, noise=20, random_state=0)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size =0.2, random_state = 0)

'''
Working with our model
'''
regressor = LinearRegression() #Creating an instance
regressor.fit(x_train,y_train) #Fitting the dataset into our model
y_pred = regressor.predict(x_test) #Predicting with the help of our dataset
loss_hist = regressor.loss_value() #Getting the list of cost function per epoch

'''
Metrics Evaluation
'''
print('R2 Score is',r2_score(y_test,y_pred))
print('Mean Square Error is',mean_squared_error(y_test,y_pred))

'''
Visualizing Our data
'''
fig,ax = plt.subplots(2,2)
#PLotting Train set vs our linear model
ax[0,0].scatter(x_train,y_train,color='red',s=0.5)
ax[0,0].plot(x_test,y_pred,color='blue')
#Plotting Test set vs our linear model
ax[0,1].scatter(x_test,y_test,color='red',s=1)
ax[0,1].plot(x_test,y_pred,color='blue')
#Plotting the loss function in each epoch
ax[1,0].plot(np.arange(500),loss_hist)
ax[1,0].set_xlabel('Epochs')
ax[1,0].set_ylabel('Loss')
#Deleting the unused Subplot
fig.delaxes(ax[1,1])
#Printing the Subplot
plt.show()