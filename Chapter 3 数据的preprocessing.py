from sklearn import preprocessing
import numpy as np 

X = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])  

# scale 零均值单位方差 
# 将数据按其属性（按列进行）减去其均值，然后除以其方差。最后得到的结果是，
# 对每个属性/每列来说所有数据都聚集在0附近，方差值为1。
X.mean(axis=0) # 用来计算数据X每个特征的均值；
X.std(axis=0) # 用来计算数据X每个特征的方差；
preprocessing.scale(X) # 直接标准化数据X
# 或者
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
# 或者 更高效！！！！！！
# 或者 更高效！！！！！！
# 或者 更高效！！！！！！
# 或者 更高效！！！！！！
# 或者 更高效！！！！！！
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)


# 最常用
# 最常用
# 最常用
# 最常用
# 最常用
# 最常用
# 最常用
# 最常用
# MinMaxScaler(最小最大值标准化)
# 公式：X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) ; 
#      X_scaler = X_std/ (max - min) + min
# 比如一个feature的取值范围远大于另一个变量，这个预处理可以让每一个feature都在-1-1之间

min_max_scaler = preprocessing.MinMaxScaler() 
X_train_minmax = min_max_scaler.fit_transform(X_train)


# MaxAbsScaler（绝对值最大标准化）
# 公式：X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) ; 
#      X_scaler = X_std/ (max) + min

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)



# 将测试数据同样应用上述数据预处理，以minmax为例
X_test = np.array([[ -3., -1., 4.]])  
X_test_minmax = min_max_scaler.transform(X_test)