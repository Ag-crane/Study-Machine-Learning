```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,6,100)   # -5~6 사이 100등분
np.random.shuffle(x)        # 순서 섞기
print(x.shape)

y = 5 + x - 2*x**2 + 1e-1*x**3 + 1e-2*x**5
y = y + 1e1*np.random.randn(100)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

plt.figure()
plt.scatter(x,y,s = 10, c = 'gray')
```

    (100,)
    




    <matplotlib.collections.PathCollection at 0x7f21f9f45520>




    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_0_2.png)
    



```python
# normal equation 이용한 regularization

N,D = x.shape

# SSE와 세타의 비율 결정
alpha = 1e1

sse = np.zeros(100)
sse[:] = np.nan

theta_norm2 = np.zeros(100)
theta_norm2[:] = np.nan

for D in range(1, 21):
    X = np.ones((N,1))
    for d in range(1, D + 1):
        X = np.hstack([X, x**d])

    # Normal equation 을 통해 구한 theta
    # alpha*np.identity(D + 1)만 추가하면 Ridge regression
    theta = np.linalg.inv(X.T.dot(X) + alpha*np.identity(D + 1)).dot(X.T).dot(y)
    y_pred = X.dot(theta)
    
    sse[D] = np.sum((y - y_pred)**2)     
    theta_norm2[D] = np.sum(theta**2)
    
    plt.figure()
    plt.scatter(x,y,s = 10, c = 'gray')
    plt.scatter(x,y_pred,s = 10, c = 'red')
    
plt.figure()
plt.plot(sse)
plt.xlabel('polynomial')
plt.ylabel('SSE cost')

plt.figure()
plt.plot(theta_norm2)
plt.ylabel('|| theta ||^2')
plt.xlabel('polynomial')
```

    <ipython-input-20-0416b63ca189>:26: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      plt.figure()
    <ipython-input-20-0416b63ca189>:31: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      plt.figure()
    




    Text(0.5, 0, 'polynomial')




    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_2.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_3.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_4.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_5.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_6.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_7.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_8.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_9.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_10.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_11.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_12.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_13.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_14.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_15.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_16.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_17.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_18.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_19.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_20.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_21.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_22.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_1_23.png)
    



```python
# sklearn 이용한 regularization

from sklearn.linear_model import LinearRegression, Ridge, Lasso

ridge_alpha = 1e1
lasso_alpha = 1e-1

linear = LinearRegression()
ridge = Ridge(alpha = ridge_alpha)
lasso = Lasso(alpha = lasso_alpha, max_iter = 1000)

X = np.ones((N,1))
for D in range(1,20):
    X = np.hstack([X, x**D])

linear.fit(X, y)
ridge.fit(X, y)
lasso.fit(X, y)

y_linear = linear.predict(X)
y_rigde = ridge.predict(X)
y_lasso = lasso.predict(X)
y_lasso = y_lasso.reshape(-1,1)

print(np.sum(linear.coef_ ** 2))
print(np.sum(ridge.coef_ ** 2))
print(np.sum(lasso.coef_ ** 2))

print(np.sum((y - y_linear)**2))
print(np.sum((y - y_rigde)**2))
print(np.sum((y - y_lasso)**2))

plt.figure()
plt.scatter(x,y,s = 10, c = 'gray')
plt.scatter(x,y_linear,s = 10, c = 'red')

plt.figure()
plt.scatter(x,y,s = 10, c = 'gray')
plt.scatter(x,y_rigde,s = 10, c = 'red')

plt.figure()
plt.scatter(x,y,s = 10, c = 'gray')
plt.scatter(x,y_lasso,s = 10, c = 'red')
```

    933.5601898037035
    7.505330332538808
    3.5926429229599743
    7142.730712570018
    7552.313491292266
    8045.23911822129
    

    C:\Users\USER\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:157: LinAlgWarning: Ill-conditioned matrix (rcond=1.21274e-29): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True, overwrite_a=True).T
    C:\Users\USER\anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.053e+03, tolerance: 6.941e+00
      model = cd_fast.enet_coordinate_descent(
    




    <matplotlib.collections.PathCollection at 0x1c8efb0e520>




    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_2_3.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_2_4.png)
    



    
![png](09_Polynomial%20Regression%20%26%20Regularization_files/09_Polynomial%20Regression%20%26%20Regularization_2_5.png)
    

