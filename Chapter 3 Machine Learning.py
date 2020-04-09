import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

# unique函数返回fruits这个表格中，fruit_label这一列中，去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
# 下面这个语句，将fruit_label fruit_name中的元素一一对应，打包成一个一一对应的元组，然后再以字典的方式输出
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   

# For this example, we use the mass, width, and height features of each fruit instance
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# -------------------------------------------通过dataset快速创建dataframe
columns = np.append(cancer.feature_names, 'target')  # 列名为数据集中feature names 和 target
index = pd.RangeIndex(start=0, stop=569, step=1)     # index 为 0/569  
data = np.column_stack((cancer.data, cancer.target)) # 数据为 cancer数据集中，data 和 target中的内容 
df = pd.DataFrame(data=data, index=index, columns=columns)  #  合并上述三项

# 快速生成一个矩阵，用来为机器学习提供输入值。下述为将-3到3分成500个数，并且形成三行三列
X_predict_input = np.linspace(-3, 3, 500).reshape(3,3)
X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1) # -1的意思是，我不知道有多少行，反正生成最后只有一列就好



# -----------------------------------------------knn 算法 
# 举例解释。比如k为3，当预测新的点时，找出训练集中离这个点最近的三个点（根据算法不同，也有可能时距离反比之和最小的点），然后根据这
# 三个点在训练集中对应的label，从而判断新的点所属的label
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# 获得鸢尾花的数据集
iris = datasets.load_iris()

# 下面是数据集切片的一种方法
x = iris.data
y = iris.target

# 测试集与训练集分离，测试集为20%的总数据
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# 对数据进行归一化处理,详情见数据的预处理
standarScaler = StandardScaler()
standarScaler.fit(X_train)

X_train_std = standarScaler.transform(X_train)
X_test_std = standarScaler.transform(X_test)

# 模型训练和测试 knn classification
knn_clf = KNeighborsClassifier(n_neighbors=5) # k = 5
# 参数详解
# 初始化函数(构造函数) 它主要有一下几个参数：
# 1.   n_neighbors=5           
# int 型参数：knn算法中指定以最近的几个最近邻样本具有投票权，默认参数为5
# 2.   weights='uniform'       
# str参数： 即每个拥有投票权的样本是按什么比重投票，'uniform'表示等比重投票，'distance'表示按距离加权
# [callable]表示自己定义的一个函数，这个函数接收一个距离数组，返回一个权值数组。默认参数为‘uniform’
# 3.   algrithm='auto'           
# str参数：即内部采用什么算法实现。有以下几种选择参数：'ball_tree':球树、'kd_tree':kd树、
# 'brute':暴力搜索、'auto':自动根据数据的类型和结构选择合适的算法。默认情况下是‘auto’。暴力搜索就不用说了大家都知道。具体前两种树型数据结构哪种好视情况而定。KD树是对依次对K维坐标轴，以中值切分构造的树,每一个节点是一个超矩形，在维数小于20时效率最高--可以参看《统计学习方法》第二章。ball tree 是为了克服KD树高维失效而发明的，其构造过程是以质心C和半径r分割样本空间，每一个节点是一个超球体。一般低维数据用kd_tree速度快，用ball_tree相对较慢。超过20维之后的高维数据用kd_tree效果反而不佳，而ball_tree效果要好，具体构造过程及优劣势的理论大家有兴趣可以去具体学习。
# 4.   leaf_size=30               
# int参数 ：基于以上介绍的算法，此参数给出了kd_tree或者ball_tree叶节点规模，
# 叶节点的不同规模会影响数的构造和搜索速度，同样会影响储树的内存的大小。具体最优规模是多少视情况而定。
# 5.   metric='minkowski'     
# str或者距离度量对象：即怎样度量距离。关于距离的介绍可以参考KNN算法解读里面的介绍，也可以参考这里。
# 6.   metric_params=None                
# 距离度量函数的额外关键字参数，一般不用管，默认为None
# 7.   n_jobs=1                
# int参数 ：指并行计算的线程数量，默认为1表示一个线程，为-1的话表示为CPU的内核数，也可以指定为其他数量的线程，
# 这里不是很追求速度的话不用管，需要用到的话去看看多线程

knn_clf.fit(X_train_std,y_train)              # 将训练集带入knn算法中，自动计算相应的参数
score = knn_clf.score(X_test_std, y_test)     # 将测试集带入建立好的模型中，测试准确性

# 预测新的数据
new_prediction = knn_clf.predict([[0.5,0.5,0.5,0.5]])

# 模型训练和测试 knn regression
regr=neighbors.KNeighborsRegressor(n_neighbors=5) # k为5的knn regression模型，其余参数类似classification
# 其余函数与classification相似  
regr.fit(X_train,y_train)


# ------------------------------------------------线性回归 y=w1x1+w2x2+w3x3+w4x4+....+wixi+b
# 利用最小二乘法，计算所有训练点实际的y值与回归函数对应的y值之间差值的平方和。找出让这个平方和最小的对应的直线的参数   

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)    
linreg = LinearRegression().fit(X_train, y_train)
linreg.coef_   # 为对应的wi
linreg.intercept_    # 为对应的b


# -------------------------------------------------岭回归，ridge 回归
# 在普通线性回归最小二乘法的基础上，加上α alpha 与所有weight wi的平方的和的乘积（L2范数），实现正则化
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

linridge = Ridge(alpha=20.0).fit(X_train, y_train)  # 设置alpha为20. alpha的值越大，正则化中，weight对回归函数的影响越大

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))                  # 输出对应的b
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))                       # 输出对应的wi
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'              # 输出测试集求出的回归模型中的准确度
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'             # 找出所有非零的wi的个数
     .format(np.sum(linridge.coef_ != 0)))





# -------------------------------------------------Lasso回归
# 在普通线性回归最小二乘法的基础上，加上α alpha 与所有weight wi的和的乘积（L1范数）（即看所有非零wi的和），实现正则化
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)              # 对数据预处理

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))                                        
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')

for e in sorted (list(zip(list(X_crime), linlasso.coef_)), key = lambda e: -abs(e[1])， reverse=False):

    # sorted函数，排序函数，key为按照什么排序。这里为按照linlasso_coef_绝对值的相反数排序。reverse若为true，为降序排序
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))                  # 输出所有wi




# --------------------------------------------------Polynomial regression
# y值由wi乘上所有feature可能构成的多项式组成。比如feature为x0 x1. 则y=w0x0+w1x1+w00x0²+w01x0x1+w11x1²+b

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 0)

poly = PolynomialFeatures(degree=2)        # 组合成的最高次数为2

X_poly = poly.fit_transform(X)       # 对数据使用这个回归之后，类似于预处理

X_train, X_test, y_train, y_test = train_test_split(X_poly, y,
                                                   random_state = 0)

linreg = Ridge().fit(X_train, y_train)     # 因为这个方法很容易导致overfitting，所以使用的时候经常加上一个正则化方法。这里是岭回归

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))






# -------------------------------------------------logistic Regression
# y的值 为，y = 1/(1+ exp[-(b+w1x1+w2x2+.......+wixi)])

from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
y_fruits_apple = y_fruits_2d == 1   # 这里使用了一个boolean，让数据集中label为1的都是苹果，其他的都不是苹果，创建了一个二元问题
X_train, X_test, y_train, y_test = (
train_test_split(X_fruits_2d.as_matrix(),
                y_fruits_apple.as_matrix(),
                random_state = 0))

clf = LogisticRegression(C=100).fit(X_train, y_train)     # 默认的正则化方法为L2范数，c为正则化的系数
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
                                         None, 'Logistic regression \
for binary classification\nFruit dataset: Apple vs others',
                                         subaxes)
                                         
new_predition = clf.predict(*)                  # 预测新的未知数据






# ------------------------------------------------- 支持向量机
# 主要调节的参数有：C、kernel、degree、gamma、coef0。
from sklearn.svm import SVC

clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None).fir(X_train,y_train)


new_prediction = clf.predict(*)
#   C：C-SVC的惩罚参数C?默认值是1.0
# C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，
# 但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。泛化能力就是对测试集的准确度，适应能力
#   kernel ：核函数，默认是rbf，可以是‘linear’,
#  ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
#   　　0 – 线性：u'v 如果数据线性可分，一般用这个 linear
#  　　 1 – 多项式：(gamma*u'*v + coef0)^degree
#   　　2 – RBF函数：exp(-gamma|u-v|^2) 如果数据线性不可分，一般用这个 radial basic function rbf
#       gamma越大，决策边界离分类的点越近，越tight
#   　　3 –sigmoid：tanh(gamma*u'*v + coef0)
#   degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
#   gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features


#   coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
#   probability ：是否采用概率估计？.默认为False
#   shrinking ：是否采用shrinking heuristic方法，默认为true
#   tol ：停止训练的误差值大小，默认为1e-3
#   cache_size ：核函数cache缓存大小，默认为200
#   class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
#   verbose ：允许冗余输出？
#   max_iter ：最大迭代次数。-1为无限制。
#   decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
#   random_state ：数据洗牌时的种子值，int值



# ---------------------------------------------------验证曲线，validation curve 
# 描述准确率与模型参数之间的关系
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(SVC(), X, y,             # 所选用的模型以及相应的数据
                                            param_name='gamma',       # 所要观察的参数名称
                                            param_range=param_range,  # 所要观察的参数的取值范围
                                            cv=3，
                                            scoring="accuracy",       # 评分方法
                                            n_jobs=1)                 # 运行的cpu个数，-1为所有


# ---------------------------------------------------学习曲线， learning curve
# 描述样本大小与测试精度和训练精度之间的关系
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                        X=X_train,
                                        y=y_train,
                                        train_sizes=np.linspace(0.1,1.0,10),
                                        cv=10,
                                        n_jobs=1)








# ---------------------------------------------------Decesion Trees 决策树

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)

clf = DecisionTreeClassifier().fit(X_train, y_train)
# 1.criterion：gini或者entropy,前者是基尼系数，后者是信息熵。
# 2.splitter： best or random 前者是在所有特征中找最好的切分点 后者是在部分特征中，默认的”best”适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐”random” 。
# 3.max_features：None（所有），log2，sqrt，N  特征小于50的时候一般使用所有的
# 4.max_depth：  int or None, optional (default=None) 设置决策随机森林中的决策树的最大深度，深度越大，越容易过拟合，推荐树的深度为：5-20之间。
# 5.min_samples_split：设置结点的最小样本数量，当样本数量可能小于此值时，结点将不会在划分。
# 6.min_samples_leaf： 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。
# 7.min_weight_fraction_leaf： 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝默认是0，就是不考虑权重问题。
# 8.max_leaf_nodes： 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
# 9.class_weight： 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。
# 10.min_impurity_split： 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值则该节点不再生成子节点。即为叶子节点 。

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))