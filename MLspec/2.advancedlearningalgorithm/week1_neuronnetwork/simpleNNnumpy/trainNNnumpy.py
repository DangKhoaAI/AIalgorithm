import numpy as np
import tensorflow as tf
import pickle
from loaddata import load_coffee_data
X,Y = load_coffee_data() 
X=np.array(X) #(200,2)
Y=np.array(Y).reshape(-1,1) #tu (200,) thanh (200,1)
#*Normalize data
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
def sigmoid(x):
    return 1/(1+np.exp(-x))
g = sigmoid
def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                  
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = g(z)               
    return(a_out)
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

#gia su sau khi train xong co du lieu
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )



#train va save model  bằng pickle

"""

# Training loop (simplified)
for epoch in range(num_epochs):
    # Forward pass
    predictions = my_sequential(X_train, W1, b1, W2, b2)
    
    # Compute loss and gradients (requires implementing)
    loss = compute_loss(y_train, predictions)
    gradients = compute_gradients(X_train, y_train, predictions, W1, b1, W2, b2)
    
    # Update weights
    W1 -= learning_rate * gradients['W1']
    b1 -= learning_rate * gradients['b1']
    W2 -= learning_rate * gradients['W2']
    b2 -= learning_rate * gradients['b2']

# Prediction
predictions = my_sequential(X_test, W1, b1, W2, b2)

# Save model parameters
with open('model_params.pkl', 'wb') as f:
    pickle.dump({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, f)
"""

#predict 
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)


X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
#them threhold
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")