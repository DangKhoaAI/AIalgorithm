import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from random import random as rd
data=pd.read_csv('D:\Project\MyProject\AIalgorithm\linear_regression\singvar\Salary_data.csv')
ye=data['YearsExperience']
sal=data['Salary']
sal=np.array(sal)/10000
#show scatter data
plt.scatter(ye,sal)
plt.xlabel('Year Experience')
plt.ylabel('Salary')
plt.xlim(0,10)
plt.ylim(3,14)
#linear regression
def minimize_value (a_now,m_now,x,y,alpha):
    da=0
    dm=0
    n=len(data.index)
    for i in range(n):
        yi=float(y[i])
        xi=float(x[i])
        da+=-(2/n)*xi*(yi-(a_now*xi+m_now))
        dm+=-(2/n)*(yi-(a_now*xi+m_now))
    a_new=a_now-da*alpha
    m_new=m_now-dm*alpha
    return a_new,m_new
a=0
m=0
iteration=5000
for i in range(iteration):
    a,m=minimize_value(a,m,ye,sal,0.01)
print(a,m)
x_func=np.linspace(0,10,11)
y_func=a*x_func+m
plt.plot(x_func,y_func)
plt.show()
