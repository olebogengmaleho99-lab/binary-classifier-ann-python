import pandas as pd
import numpy as np

#import the dataset
df = pd.read_csv('/Users/olebogengmaleho/Desktop/Datasets/house-votes-84.data.csv',na_values=['?'])
#handling missing values and encoding categorial variables
df.dropna(inplace=True)
pd.set_option('future.no_silent_downcasting', True)
df.replace(('y', 'n'), (1, 0), inplace=True)
df.replace(('democrat', 'republican'), (1.0, 0.0), inplace=True)
#converting pandas dataframe to numpy array
data = np.array(df)
m, n = data.shape
#shuffling and splitting dataset into training and testing data
np.random.shuffle(data)
train = data[0:186].T
test = data[186:].T
X_train = train[1:n] #features
Y_train = train[0] #labels
X_test = test[1:n]
Y_test = test[0]

def init_params():
    W1 = np.random.rand(10,16) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(1,10) - 0.5
    b2 = np.random.rand(1,1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0,Z)

def ReLU_deriv(Z):
    return Z > 0

def sigmoid(Z):
    Z = Z.astype(np.float64)    
    A = 1.0 / (1.0 + np.exp(-Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
	
def backward_prop(A1,A2,W2,X,Y,Z1):
    BCE = -np.mean(Y*np.log(A2)+(1-Y)*np.log(1-A2))
    dW2 = 1 / m * (A2 - Y).dot(A1.T)
    db2 = 1 / m * np.sum(A2 - Y)
    dZ1 = W2.T.dot(A2 - Y) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum((W2.T).dot(A2 - Y)*ReLU_deriv(Z1))
    return dW1, db1, dW2, db2

def gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    for i in range(A2.size):
        if A2[0,i] >= 0.5:
           A2[0,i] = 1.0
        else: 
            A2[0,i] = 0.0             
    return A2

def get_accuracy(ypred,ytrue):
	n = ytrue.size
    total = 0.0
    for i in range(n):
        if ypred[0,i] == ytrue[i]:
           total = total + 1 
    return ((total / n) * 100)
        
W1,b1,W2,b2 = init_params()
#training the model
for i in range(10000):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
    dW1, db1, dW2, db2 = backward_prop(A1,A2,W2,X_train,Y_train,Z1)
    W1, b1, W2, b2 = gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.001)

#testing model performance on unseen data
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_test)
Accuracy = get_accuracy(get_predictions(A2),Y_test)
print(f"Model accuracy is {Accuracy} %")






