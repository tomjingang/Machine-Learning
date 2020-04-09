def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    # Your code here
    x_predict = np.linspace(0,10,100)
    result = np.zeros((4,100))                     # 构建储存数据的数组
#     for i in [1,3,6,9]:
    degree = [1,3,6,9]
    for i,j in enumerate(degree):                  # i为0，1，2，3的索引list， j为degree中的对应元素
        poly = PolynomialFeatures(degree=j)       

        X_poly = poly.fit_transform(x.reshape(-1,1)) # 一定要注意，在sklearn中，一定是二维的向量，所以当只有一个feature时，需要加
                                                      # reshape将数据变成二维数组


        X_train, X_test, y_train, y_test = train_test_split(X_poly, y,random_state = 0)
        
        
        linreg = LinearRegression().fit(X_train, y_train)
        prediction = linreg.predict(poly.fit_transform(x_predict.reshape(100,1)))
        result[i,:] = prediction

        y_pre_test_lin = linreg.predict(X_test)   # 计算对应的r2_score      
        
        lin_r2 = r2_score(y_test, y_pre_test_lin)
        
    return result
