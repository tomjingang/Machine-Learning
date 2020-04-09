# 回归模型随机数据 X为样本特征， y为样本输出

from sklearn.datasets.samples_generator import make_regression

X,y, coef = make_regression(n_samples=100, # 样本数

    n_features=100,  # 特征数

    n_informative=10, # 提供有用信息的特征数

    n_targets=1, # 输出的target的维数

    bias=0.0,  # 线性模型的偏差率, 比如说z=wx+b, b就是bias

    effective_rank=None,

    noise=0.0, # 线性模型的噪声

    shuffle=True, # 是否将特征和抽样洗牌

    coef=False, # 是否返回相关系数

    random_state=None)



# 分类模型随机数据 X1为样本特征，Y1为样本输出
from sklearn.datasets.samples_generator import make_classification

X1, Y1 = make_classification(n_samples=400, # 样本数

    n_features=2, # 特征数

    random_state = 0, # 随机核

    n_informative = 2, # 有效的特征数

    flip_y = 0.1, # 标签反转的概率

    class_sep = 0.5, 
    
    n_redundant=0, # 没有用的特征数
                             
    n_clusters_per_class=1, # 每个类别有几个簇
    
    n_classes=3 # 输出的类别数
    )



# 正态分布混合数据

from sklearn.datasets import make_gaussian_quantiles
#生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
X1, Y1 = make_gaussian_quantiles(n_samples=1000, 

        n_features=2, # 正态分布维数

        n_classes=3, # 类别数

        mean=[1,2],  # 每一维度的均值

        cov=2)       # 协方差系数