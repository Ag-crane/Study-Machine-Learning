```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import svm

from keras.datasets import mnist

#  28 x 28 pixel로 이루어진 손글씨 이미지 => 784 x 1짜리 벡터로 변환
((x_train, y_train), (x_test, y_test)) = mnist.load_data()

X_train = x_train.reshape(-1,28*28)
X_test = x_test.reshape(-1,28*28)

N_train,D = X_train.shape
N_test,D = X_test.shape

# dimensional reduction (PCA) : 시간 단축하려고
# covariance
mu = np.mean(X_train, axis = 0)
C = X_train - mu
C = (1/N_train) * C.T.dot(C)

# SVD: C = U*S*V^T
U, s, V = np.linalg.svd(C, full_matrices = True)
S = np.diag(s)
V = V.T

# 전체 입력 차원 784 중에서, 차원을 축소해서 사용
D_reduce = 30
X_train = X_train.dot(V[:,0:D_reduce]) # N_train x D_reduce
X_test = X_test.dot(V[:,0:D_reduce]) # N_train x D_reduce

X_train = np.hstack([np.ones((N_train,1)),X_train])
X_test = np.hstack([np.ones((N_test,1)),X_test])

# one-of-K coding
num = np.unique(y_train, axis = 0)
num = num.shape[0]

t_train = np.eye(num)[y_train]
t_test = np.eye(num)[y_test]
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 1s 0us/step
    


```python
print(X_train.shape)
print(N_train)
print(t_train.shape)
```

    (60000, 31)
    60000
    (60000, 10)
    


```python
# parameter
M = np.array([X_train.shape[1],100,1000,10])  # layer별 뉴런 개수. 시작은 input+1(bias), 마지막은 클래스 개수
L = M.shape[0] - 1   # layer 수  

eta = 1e-5
maxEpoch = 30

# initialize parameter
# W[0] = 31 x 100
# W[1] = 100 x 1000
# W[2] = 1000 x 10
W = []

for l in range(L):
    W.append(np.random.randn(M[l],M[l + 1]))
    print(W[l].shape)
```

    (31, 100)
    (100, 1000)
    (1000, 10)
    


```python
# activation function
def act(x):        # activation func 
    return np.tanh(x)

def dact(x):       # 도함수
    return 1 - np.tanh(x)**2

# softmax
def softmax(x):    
    
    if x.ndim == 1:
        f_x = np.exp(x)
        return f_x / np.sum(f_x)
    
    elif x.ndim == 2:
        max = np.max(x,axis = 1,keepdims = True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x,axis = 1,keepdims = True)
        f_x = e_x / sum 
        return f_x

# cross entropy
def cross_entropy(y,t):
    N,K = y.shape
    e = np.sum(- t * np.log(y), axis = 1)
    e = np.mean(e)
    
    return e

cost = []
accuracy = []

# initialize for a single input

# forward propagation
# 뉴런 개수만큼
z_sample = []
a_sample = []
for l in range(L):    
    z_sample.append(np.zeros((1,M[l + 1])))
    a_sample.append(np.zeros((1,M[l + 1])))    
pred_sample = softmax(z_sample[-1])   # output y  

# backpropagation
delta = []
for l in np.arange(L):    
    delta.append(np.zeros((M[l],1)))
        
# initialize for entire inputs
# forward propagation
z_train = []
a_train = []
for l in range(L):     # N개 한번에 계산할 준비  
    z_train.append(np.zeros((N_train,M[l + 1])))
    a_train.append(np.zeros((N_train,M[l + 1])))    
pred_train = softmax(z_train[-1])    
```


```python
# stochastic gradient descent
for epoch in range(maxEpoch):  
    # 학습 데이터의 랜덤 순서로 사용할 '랜덤 순열'
    N_perm = np.random.permutation(N_train)    
    
    for n in range(N_train):        
        W_new = W
        
        ### forward propagation
        # input layer
        z_sample[0] = X_train[N_perm[n],:].dot(W[0])
        a_sample[0] = act(z_sample[0])

        # 나머지 layer
        for l in range(L - 1):    
            z_sample[l + 1] = a_sample[l].dot(W[l + 1])
            a_sample[l + 1] = act(z_sample[l + 1])

        # output layer
        pred_sample = softmax(z_sample[-1])
    
        ### backpropagation : 각 layer마다 delta, parameter 구함
        # output layer
        delta[L - 1] = -(t_train[N_perm[n],:] - pred_sample)  # local gradient

        # propagation
        for l in np.arange(L - 1,1,-1):
            delta[l - 1] = W[l].dot(delta[l]) * dact(z_sample[l - 1])         
        
        # input layer        
        delta[0] = W[1].dot(delta[1]) * dact(z_sample[0]) 

        ### update
        W_new[L - 1] = W_new[L - 1] - eta*a_sample[L - 2].reshape(-1,1).dot(delta[L - 1].reshape(1,-1))
        for l in np.arange(L - 1,1,-1):
            W_new[l - 1] = W_new[l - 1] - eta*a_sample[l - 2].reshape(-1,1).dot(delta[l - 1].reshape(1,-1))  
        # 마지막은 a 대신 x
        W_new[0] = W_new[0] - eta*X_train[N_perm[n],:].reshape(-1,1).dot(delta[0].reshape(1,-1))  
        
        W = W_new

    # forward propagation
    z_train[0] = X_train.dot(W[0])
    a_train[0] = act(z_train[0])

    for l in range(L - 1):    
        z_train[l + 1] = a_train[l].dot(W[l + 1])
        a_train[l + 1] = act(z_train[l + 1])

    pred_train = softmax(z_train[-1])

    # performance
    cost.append(cross_entropy(pred_train,t_train))
    accuracy.append(np.sum(y_train == np.argmax(pred_train, axis = 1)) / N_train)
    
    print('[epoch %d] cross entropy: %.4f, accuracy: %.4f'%(epoch,cost[-1],accuracy[-1]))     
```

    [epoch 0] cross entropy: 21.8667, accuracy: 0.2765
    [epoch 1] cross entropy: 15.1697, accuracy: 0.4118
    [epoch 2] cross entropy: 12.9498, accuracy: 0.4566
    [epoch 3] cross entropy: 10.5941, accuracy: 0.5310
    [epoch 4] cross entropy: 9.4110, accuracy: 0.5677
    [epoch 5] cross entropy: 9.0384, accuracy: 0.5838
    [epoch 6] cross entropy: 8.2591, accuracy: 0.6067
    [epoch 7] cross entropy: 7.8879, accuracy: 0.6072
    [epoch 8] cross entropy: 7.7324, accuracy: 0.6153
    [epoch 9] cross entropy: 7.1278, accuracy: 0.6431
    [epoch 10] cross entropy: 7.1726, accuracy: 0.6427
    [epoch 11] cross entropy: 6.8858, accuracy: 0.6519
    [epoch 12] cross entropy: 6.4602, accuracy: 0.6721
    [epoch 13] cross entropy: 6.1490, accuracy: 0.6784
    [epoch 14] cross entropy: 6.3072, accuracy: 0.6697
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-5-84f1aac2b053> in <module>
         26         # propagation
         27         for l in np.arange(L - 1,1,-1):
    ---> 28             delta[l - 1] = W[l].dot(delta[l]) * dact(z_sample[l - 1])
         29 
         30         # input layer
    

    KeyboardInterrupt: 



```python
plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(cost)
plt.xlabel('epoch')
plt.ylabel('cross entropy')
plt.subplot(2,1,2)
plt.plot(accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
```


```python
# 똑같은걸 keras로 만들기. 훨씬 빠름

import keras
from keras.models import Sequential
from keras.layers import Dense

num_classes = 10
epochs = maxEpoch

opt = keras.optimizers.SGD(learning_rate=eta, name="SGD")  # stochastic gradient decent

model = Sequential()
# Dense : fully connected layer
model.add(Dense(100, activation='tanh', input_dim = X_train.shape[1], use_bias = False, kernel_initializer='random_normal'))
model.add(Dense(1000, activation='tanh', use_bias = False, kernel_initializer='random_normal'))
model.add(Dense(num_classes, activation='softmax', use_bias = False, kernel_initializer='random_normal'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
hist = model.fit(X_train, t_train, epochs = epochs, verbose = 1)
```
