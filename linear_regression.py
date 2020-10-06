import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = None
        self.weights = None
        self.bias = None
        self.loss_history = []

    #Method to fit into SimpleLinear Regression
    def fit(self,X,y):
        #initilizae our parametes
        n_samples, n_features = X.shape
        self.loss = self.weights  = np.zeros(n_features)
        self.bias = 0

        #Implementing Gradient Descent
        for _ in range(self.epochs):
            #This is our Linear Eqaution of the model
            y_pred = np.dot(X, self.weights) + self.bias
                    
            #Calculating the loss(cost function) of the model
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y)**2) 
            self.loss_history.append(cost)

            #Calculating the gradient for weight and biases
            grad_weight = (1/n_samples) * np.dot(X.T,(y_pred - y)) 
            grad_bias = (1/n_samples) * np.sum(y_pred - y)

            #Updating the weights and bias
            self.weights -= self.learning_rate * grad_weight
            self.bias -= self.learning_rate * grad_bias
            
    #Method to predict the target
    def predict(self,X):
         y_pred = np.dot(X, self.weights) + self.bias
         return y_pred

    
    #Method to perform after Fit Method to get loss function
    def loss_value(self):
        return self.loss_history

    

    



