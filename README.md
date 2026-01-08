# Algorithm for QR Decomposition
## Aim:
To implement QR decomposition algorithm using the Gram-Schmidt method.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
### Step1

Import the necessary Python libraries such as NumPy, Matplotlib, and Scikit-learn.

### Step 2

Load the dataset using the datasets module and store the input features in X and the output values in y.

### Step 3

Split the dataset into training data and testing data using train_test_split().

### Step 4

Create a Linear Regression model using LinearRegression().

### Step 5

Train the model using the training data and predict the output.
Evaluate the model using the variance score and plot the residual errors.

## Program:
### Gram-Schmidt Method
```
''' 
Program to QR decomposition using the Gram-Schmidt method
Developed by:KNOWKASH G
RegisterNumber:25015209 
'''
import numpy as np

def QR_decomposition(A):
    A=np.array(A,dtype=float)
    m,n=A.shape
    
    Q,R=np.zeros((m,n)),np.zeros((n,n))

    for j in range(n):
        v=A[:,j]
        for i in range(j):
            R[i,j]=np.dot(Q[:,i],A[:,j])
            v-=R[i,j]*Q[:,i]
            
        R[j,j]=np.linalg.norm(v)
        Q[:,j]=v/R[j,j]
    print("The Q Matrix is \n",Q) 
    print("The R Matrix is \n",R) 
    
a=np.array(eval(input()))
QR_decomposition(a)

```

## Output
<img width="1173" height="821" alt="image" src="https://github.com/user-attachments/assets/6364ce8a-ecdb-4500-9863-b6b0f04e894a" />

## Result
Thus the QR decomposition algorithm using the Gram-Schmidt process is written and verified the result.
