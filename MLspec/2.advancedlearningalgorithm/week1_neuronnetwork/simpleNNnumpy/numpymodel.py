import numpy as np
import tensorflow as tf
from loaddata import load_coffee_data
X,Y = load_coffee_data() 
X=np.array(X) #(200,2)
Y=np.array(Y).reshape(-1,1) #tu (200,) thanh (200,1)
#*Normalize data
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
norm_l.adapt(X)
Xn = np.array(norm_l(X))
#*nhan ban du lieu
"""np.tile(array, reps) là hàm của NumPy, dùng để "lặp" một mảng theo một số lần nhất định"""
#//Xt = np.tile(Xn,(1000,1)) #200000,2
#//Yt= np.tile(Y,(1000,1))   #200000,1
print(Xn.shape,Y.shape)
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
class MyModel:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def my_dense(self, a_in, W, b):
        units = W.shape[1]
        a_out = np.zeros(units)
        for j in range(units):
            w = W[:,j]
            z = np.dot(w, a_in) + b[j]
            a_out[j] = self.sigmoid(z)
        return a_out

    def my_sequential(self, x):
        a1 = self.my_dense(x, self.W1, self.b1)
        a2 = self.my_dense(a1, self.W2, self.b2)
        return a2

    def compile(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # Define the cost function and gradient computation for logistic regression
        def compute_cost_logistic(y, y_pred):
            m = y.shape[0]
            cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            return cost

        def compute_gradient_logistic(X, y, y_pred):
            m = X.shape[0]
            gradient = (1 / m) * np.dot(X.T, (y_pred - y))
            return gradient

        self.compute_cost_logistic = compute_cost_logistic
        self.compute_gradient_logistic = compute_gradient_logistic
        print("Model compiled with logistic regression loss and gradient.")
    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.num_iterations):
            y_pred = self.sigmoid(np.dot(X, self.theta))
            cost = self.compute_cost_logistic(y, y_pred)
            gradient = self.compute_gradient_logistic(X, y, y_pred)
            self.theta -= self.learning_rate * gradient

        print(f"Training completed. Final cost: {cost:.4f}")

    def predict(self, x):
        return self.my_sequential(x)
    
    def save(self, filename):
        # Lưu model vào file dưới dạng numpy (hoặc bạn có thể sử dụng pickle)
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to {filename}.npz")

    @staticmethod
    def load(filename):
        # Load model từ file
        data = np.load(filename + ".npz")
        return MyModel(data['W1'], data['b1'], data['W2'], data['b2'])

# khởi tạo trọng số 
W1 = np.random.rand(2, 3) # Giả sử input có 2 phần tử, output 3 units
b1 = np.random.rand(3)
W2 = np.random.rand(3, 1)  # Giả sử output cuối là 2 units
b2 = np.random.rand(1)
#khoi tạo model 
model = MyModel(W1, b1, W2, b2)
model.compile()
#huấn luyện model
model.fit(Xn,Y)
x = np.random.rand(3)  # input 3 phần tử
y_pred = model.predict(x)
print("Predicted:", y_pred)
 
#*predict 
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

#đưa từ xác suất sang 0/1
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
"""
# Lưu model
model.save('my_model')

# Tải model
loaded_model = MyModel.load('my_model')
y_pred_loaded = loaded_model.predict(x)
print("Predicted by loaded model:", y_pred_loaded)
"""
