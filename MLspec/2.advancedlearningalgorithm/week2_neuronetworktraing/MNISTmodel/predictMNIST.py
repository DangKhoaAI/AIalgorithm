import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from loaddata import load_data
model=load_model("MNISTmodel.h5")
X,Y=load_data(filepath="MNISTdata.csv")
numtest=300
sample_indices = X.sample(n=numtest, random_state=42).index
Xtest = np.array(X.loc[sample_indices])
Ytest = np.array(Y.loc[sample_indices])
#*predict
predict=model.predict(Xtest)
count=0
for i in range(numtest):
    print( f"yhat: {np.argmax(predict[i])}, y={Ytest[i]}")
    if np.argmax(predict[i])==Ytest[i]:
        count+=1
print(f"Accuracy of model in test is:{(count/len(Ytest))*100} %")