# train_test_split
# 返回切分的数据集train/test
train_test_split(*array,test_size=0.25,train_size=None,random_state=None,shuffle=True,stratify=None)1
# *array：切分数据源（list/np.array/pd.DataFrame/scipy_sparse matrices） 
# test_size和train_size是互补和为1的一对值 
# shuffle：对数据切分前是否洗牌 
# stratify：是否分层抽样切分数据（ If shuffle=False then stratify must be None.） 
# random_state 是这个随机分组的random seed
# 举例
import numpy as np 
from sklearn.model_selection import train_test_split

X, y = (np.arange(10).reshape((5,2)),range(5))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40, random_state=30)




#cross_validate
# 返回train/test数据集上的每折得分
cross_validate(estimator,X,y=None,groups=None,scoring=None,cv=None,n_jobs=1,
verbose=0,fit_params=None,pre_dispatch='2*n_jobs',return_train_score='warn')12
# estimator：学习器 
# X：特征列数据 
# y：标签列（无监督学习可以无此参数） 
# groups：切分train/test数据集后的样本所在集合标号 
# scoring：在test数据集上的评估准则（以list/dict形式给出） 
# cv：交叉验证的折数，default=3，也可以是其余int数据，或者cv generator 
# n_jobs：计算执行时占用CPU个数，设置n_jobs=-1是利用全部CPU 
# verbose：设置评估模型的相关打印信息输出详细程度 
# fit_params：参数字典 
# pre_dispatch：设置并行任务数（保护内存） 
# return_train_score：返回train数据集上的评估得分



#GridSearchCV
# 返回最佳参数组合/得分

GridSearchCV(estimator,para_grid,scoring=None,n_jobs=1,iid=True,refit=True,cv=None,
verbose=0,pre_dispatch='2*n_jobs',error_score='raise',return_train_score='warn')12
# estimator：学习器 
# para_grid：参数字典 
# scoring：在test数据集上的评估准则（以list/dict形式给出） 
# n_jobs：计算执行时占用CPU个数，设置n_jobs=-1是利用全部CPU 
# iid：是否假设样本同分布，建模时目标函数时计入每个样本的总损失 
# cv：交叉验证的折数，default=3，也可以是其余int数据，或者cv generator 
# verbose：设置评估模型的相关打印信息输出详细程度 
# pre_dispatch：设置并行任务数（保护内存） 
# return_train_score：返回train数据集上的评估得分 
# error_score：设置estimator拟合出现错误时的相关提示信息，对refit有影响 
# refit：利用最优参数组合做什么？待研究 
# （refit : boolean, or string, default=True 
# Refit an estimator using the best found parameters on the whole dataset. 
# For multiple metric evaluation, this needs to be a string denoting the scorer is used to find the best parameters for refitting the estimator at the end. 
# The refitted estimator is made available at the best_estimator_ attribute and permits using predict directly on this GridSearchCV instance. 
# Also for multiple metric evaluation, the attributes best_index_, best_score_ and best_parameters_ will only be available if refit is set and all of them will be determined w.r.t this specific scorer. 
# See scoring parameter to know more about multiple metric evaluation.） 






learning_curve
根据设定的不同train数据集大小,依次获得交叉验证的train/test数据集上的得分



GridSearchCV(estimator,X,y,groups=None，train_sizes=array([0.1,0.33,0.55,0.78,1.]),cv=None,
scoring=None,exploit_incremental_learning=False,n_jobs=1,
pre_dispatch='all',verbose=0,shuffle=False,random_state=None)123

# estimator：学习器 
# X：特征列数据 
# y：标签列 
# groups：切分train/test数据集后的样本所在集合标号 
# train_sizes：设置训练集数据的变化取值范围 
# cv：交叉验证的折数，default=3，也可以是其余int数据，或者cv generator 
# scoring：在test数据集上的评估准则（以list/dict形式给出） 
# n_jobs：计算执行时占用CPU个数，设置n_jobs=-1是利用全部CPU 
# pre_dispatch：设置并行任务数（保护内存） 
# verbose：设置评估模型的相关打印信息输出详细程度 
# shuffle：对数据切分前是否洗牌 
# random_state：随机种子 
# exploit_incremental_learning：增量学习







# --------------------------------------------------- cross validation
from sklearn.model_selection import cross_val_score

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits_2d.as_matrix()
y = y_fruits_2d.as_matrix()
cv_scores = cross_val_score(clf, X, y)






