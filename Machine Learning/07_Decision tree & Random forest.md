```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import svm

from keras.datasets import mnist

# 28 x 28 pixel로 이루어진 손글씨 이미지 train : 60000개, test : 10000개
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 60000개의 784x1 벡터로 변환
X_train = x_train.reshape(60000, 28*28)  
X_test = x_test.reshape(10000, 28*28)
print(X_train.shape, X_test.shape)

# # 맨 첫 번째 이미지
# plt.figure()
# print(x_train[0,:,:])  # 0~255 정수 28x28
# plt.imshow(x_train[0,:,:], cmap = 'gray')  # 를 그림으로 그리면 (히트맵)
```

    (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    (60000, 784) (10000, 784)
    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136
      175  26 166 255 247 127   0   0   0   0]
     [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253
      225 172 253 242 195  64   0   0   0   0]
     [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251
       93  82  82  56  39   0   0   0   0   0]
     [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119
       25   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253
      150  27   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252
      253 187   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249
      253 249  64   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
      253 207   2   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253
      250 182   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201
       78   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]]
    




    <matplotlib.image.AxesImage at 0x7f08e958e910>




    
![png](07_Decision%20tree%20%26%20Random%20forest_files/07_Decision%20tree%20%26%20Random%20forest_0_2.png)
    



```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 

max_cluster = 25

score_inertia = np.full(max_cluster + 1, np.nan)     # nan으로 채워진 26x1 벡터 n
score_silhouette = np.full(max_cluster + 1, np.nan)

for n_cluster in range(2, max_cluster + 1):
    kmeans = KMeans(n_clusters = n_cluster, random_state = 0).fit(X_test)  # 너무 오래걸려서 test셋으로 돌림
    score_inertia[n_cluster] = kmeans.inertia_   # 구해져있음
    score_silhouette[n_cluster] = silhouette_score(X_test, kmeans.labels_)
    
plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(score_inertia,'o-')
plt.subplot(2,1,2)
plt.plot(score_silhouette,'o-')
```




    [<matplotlib.lines.Line2D at 0x7f08e9525550>]




    
![png](07_Decision%20tree%20%26%20Random%20forest_files/07_Decision%20tree%20%26%20Random%20forest_1_1.png)
    



```python
from sklearn import tree
from sklearn.metrics import accuracy_score

model_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 1000, min_samples_leaf = 1000)
model_tree.fit(X_train, y_train)

y_tree_train = model_tree.predict(X_train)
y_tree_test = model_tree.predict(X_test)

print( 'train %.4f, test %.4f' % (accuracy_score(y_tree_train, y_train), accuracy_score(y_tree_test, y_test)) )
```

    train 0.6691, test 0.6740
    


```python
# decision tree를 시각화
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

dot_data = export_graphviz(model_tree, out_file=None, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
```




    
![png](07_Decision%20tree%20%26%20Random%20forest_files/07_Decision%20tree%20%26%20Random%20forest_3_0.png)
    




```python
from sklearn.ensemble import RandomForestClassifier

numlist_ensemble = np.array([1, 2, 5, 10, 20, 50, 100])
accuracy_train = []
accuracy_test = []

for num_ensemble in numlist_ensemble:
    model_rdf = RandomForestClassifier(n_estimators = num_ensemble, max_depth = 10, min_samples_split = 1000, min_samples_leaf = 1000)
    model_rdf.fit(X_train, y_train)

    y_rdf_train = model_rdf.predict(X_train)
    y_rdf_test = model_rdf.predict(X_test)

    print( 'ensemble %d: train %.4f, test %.4f' % (num_ensemble, accuracy_score(y_rdf_train, y_train), accuracy_score(y_rdf_test, y_test)) )

    accuracy_train.append(accuracy_score(y_rdf_train, y_train))
    accuracy_test.append(accuracy_score(y_rdf_test, y_test))
    
plt.figure()
plt.plot(numlist_ensemble, accuracy_train, 'o-')
plt.plot(numlist_ensemble ,accuracy_test, 'o-')
```

    ensemble 1: train 0.5386, test 0.5507
    ensemble 2: train 0.6350, test 0.6444
    ensemble 5: train 0.7479, test 0.7569
    ensemble 10: train 0.7925, test 0.7918
    ensemble 20: train 0.8206, test 0.8248
    ensemble 50: train 0.8497, test 0.8528
    ensemble 100: train 0.8475, test 0.8517
    




    [<matplotlib.lines.Line2D at 0x2ac4ccc2a00>]




    
![png](07_Decision%20tree%20%26%20Random%20forest_files/07_Decision%20tree%20%26%20Random%20forest_4_2.png)
    

