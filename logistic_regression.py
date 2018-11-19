# Import libraries
import numpy as np
import math
from sklearn.model_selection import train_test_split


#Define logistic regression function
def logistic_regression(X_train, X_test, Y_train, Y_test, betas):
    # Get the size of the attrivute array and no of train data points	
    n_attr = X_train.shape[1]
    n_train = X_train.shape[0]
    # Probability
    p = np.zeros(n_train)
    r = np.zeros(n_train)
    for i in range(n_train):
        # Calculating the probability term using sigmoid function
        exp_temp= math.exp(betas[0] + sum(np.multiply(betas[1:], X_train[i,])))
        p[i] = exp_temp/(1 + exp_temp)
        # Calculating the error terms for each obs in training set
        r[i] = (Y_train[i] - p[i])**2.0
   
    # MSE 
    mse = np.sum(r)/n_train
    itr=0
    
    mse_to_be_checked = mse 
    # Compute the derivatives
    while(mse>0.05):
        itr+=1
        d = np.zeros(n_train)
        for i in range(n_train):
            exp_temp= math.exp(betas[0] + sum(np.multiply(betas[1:], X_train[i,])))
            sigmoid = exp_temp/(1 + exp_temp)
            sigmoid_d = exp_temp/((exp_temp + 1)**2)
            d[i] = 2*np.subtract(Y_train[i], p[i])*sigmoid_d
    
        #learning_rate, one should try for different learning rates to find optimal value of rates
        eta = 1
        for j in range(betas.shape[0]):
            for i in range(n_train):
                if j==0:
                    betas[j] = betas[j] + (eta*d[i]/n_train)
                else: 
                    betas[j] = betas[j] + (eta*d[i]*X_train[i,j-1]/n_train)
        for i in range(n_train):
            # Calculating the probability term using sigmoid function
            exp_temp= math.exp(betas[0] + sum(np.multiply(betas[1:], X_train[i,])))
            p[i] = exp_temp/(1 + exp_temp)
            # Calculating the error terms for each obs in training set
            r[i] = (Y_train[i] - p[i])**2.0
   
        # MSE 
        # Stopping the loop if mse doesnt change by 1% in 10 itrs. 
        mse = np.sum(r)/n_train
        if itr%10 == 0:
            if mse_to_be_checked - mse < 0.01:
                break
            else:
                mse_to_be_checked = mse
    n_test = X_test.shape[0]
    mse_test =0
    for i in range(n_test):
        #Calculating the probability term using sigmoid function
        exp_temp= math.exp(betas[0] + sum(np.multiply(betas[1:], X_test[i,])))
        temp = exp_temp/(1 + exp_temp)
        mse_test = mse_test + (Y_test[i]- temp)**2
    mse_test = mse_test/n_test
    
    print("The train MSE is %f for %d iteration and learning rate is set at %f. \n"%(mse, itr, eta))
    print("The test MSE is: ",mse_test, "\n\n")
    return mse_test


# Import data from a file
data = np.genfromtxt("auto.data", skip_header = 1, usecols = (0,3,4,6,7))
#Drop the data points with missing data
data = data[~np.isnan(data).any(axis=1)]
# Create a binary label from present label
auto_Y = (data[:,0]>=23)*1
auto_X = data[:,[1,2,3,4]]

# Standardize the data
auto_X[:,0] = (auto_X[:,0] - np.mean(auto_X[:,0]))/np.std(auto_X[:,0])
auto_X[:,1] = (auto_X[:,1] - np.mean(auto_X[:,1]))/np.std(auto_X[:,1])
auto_X[:,2] = (auto_X[:,2] - np.mean(auto_X[:,2]))/np.std(auto_X[:,2])
auto_X[:,3] = (auto_X[:,3] - np.mean(auto_X[:,3]))/np.std(auto_X[:,3])

# Split the data into test and train params and labels.
X_train, X_test, Y_train, Y_test = train_test_split(auto_X, auto_Y, random_state=1109)

#Training the algorithm with different learning rates
betas = np.random.rand(auto_X.shape[1] + 1)*1.4 - 0.7

# Run the function over the data
logistic_regression(X_train, X_test, Y_train, Y_test, betas)


