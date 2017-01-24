在https://github.com/letiantian/dataset下载text-classification.7z，解压后导入数据：

$ ls
test_data.npy  test_labels.npy  training_data.npy  training_labels.npy
$ ipython
>>> import numpy as np
>>> training_data = np.load("training_data.npy")
>>> training_data.shape
(1998, 19630)
>>> training_labels = np.load("training_labels.npy")
>>> training_labels
array([6, 6, 6, ..., 2, 2, 2])
>>> training_labels.shape
(1998,)
>>> test_data = np.load("test_data.npy")
>>> test_data.shape
(509, 19630)
>>> test_labels = np.load("test_labels.npy")
>>> test_labels.shape
(509,)

使用多项式贝叶斯

>>> from sklearn.naive_bayes import MultinomialNB
>>> clf =MultinomialNB()
>>> clf.fit(training_data, training_labels)  # 训练模型
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
>>> predict_labels = clf.predict(test_data)  # 预测训练集
>>> sum(predict_labels == test_labels)       # 预测对了几个？
454
>>> len(predict_labels)                      # 训练样本个数
509
>>> 454./509                                 # 正确率
0.8919449901768173                           # 效果不错
>>> from sklearn.metrics import confusion_matrix 
>>> confusion_matrix(test_labels, predict_labels)  # 混淆矩阵
array([[ 39,   0,   0,   1,   0,   1,   0,   0], 
       [  0,  32,   1,   0,   0,   4,   0,   1],
       [  0,   0,  50,   0,   0,   8,   0,   4],
       [  0,   0,   1,  44,   0,  10,   0,   0],
       [  1,   0,   0,   0,  66,   0,   0,   1],
       [  2,   2,   1,   6,   1, 144,   1,   1],
       [  0,   0,   0,   0,   0,   2,  25,   0],

使用伯努利贝叶斯

>>> from sklearn.naive_bayes import BernoulliNB
>>> clf2 = BernoulliNB()
>>> clf2.fit(training_data, training_labels)  # 训练模型
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
>>> predict_labels = clf2.predict(test_data)  # 预测训练集
>>> sum(predict_labels == test_labels)        # 预测对了几个？
387
>>> 387./509                                  # 正确率
0.7603143418467584
这个和下面的效果是一样的：

>>> clf2 = BernoulliNB()
>>> clf2.fit(training_data>0, training_labels)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
>>> predict_labels = clf2.predict(test_data>0)
>>> sum(predict_labels == test_labels)
387
使用高斯贝叶斯

>>> from sklearn.naive_bayes import GaussianNB
>>> clf3 = GaussianNB()
>>> clf3.fit(training_data, training_labels)   # 训练模型
GaussianNB()
>>> predict_labels = clf3.predict(test_data)   # 预测训练集
>>> sum(predict_labels == test_labels)         # 预测对了几个？
375
>>> 375./509                                   # 正确率
0.7367387033398821




import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]])
print "enc.n_values_ is:",enc.n_values_
print "enc.feature_indices_ is:",enc.feature_indices_
print enc.transform([[0, 1, 1]]).toarray()

enc.n_values_ is: [2 3 4]
enc.feature_indices_ is: [0 2 5 9]
[[ 1.  0.  0.  1.  0.  0.  1.  0.  0.]]


>>> from sklearn.ensemble import GradientBoostingClassifier
>>> gbdt = GradientBoostingClassifier()
>>> gbdt.fit(training_data, training_labels)  # 训练。喝杯咖啡吧
GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',  
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
>>> gbdt.feature_importances_   # 据此选取重要的特征
array([  2.08644807e-06,   0.00000000e+00,   8.93452010e-04, ...,  
         5.12199658e-04,   0.00000000e+00,   0.00000000e+00])
>>> gbdt.feature_importances_.shape
(19630,)


>>> gbdt_predict_labels = gbdt.predict(test_data)
>>> sum(gbdt_predict_labels==test_labels)  # 比 多项式贝叶斯 差许多
414 

>>> new_train_data = training_data[:, feature_importances>0]
>>> new_train_data.shape  # 只保留了1636个特征
(1998, 1636)
>>> new_test_data = test_data[:, feature_importances>0]
>>> new_test_data.shape
(509, 1636)



>>> from sklearn.naive_bayes import MultinomialNB
>>> bayes = MultinomialNB() 
>>> bayes.fit(new_train_data, training_labels)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  
>>> bayes_predict_labels = bayes.predict(new_test_data)
>>> sum(bayes_predict_labels == test_labels)   # 之前预测正确的样本数量是454
445 


对原始特征组成的数据集：

>>> from sklearn.linear_model import LogisticRegression
>>> lr1 = LogisticRegression()
>>> lr1.fit(training_data, training_labels)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,  
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0)
>>> lr1_predict_labels = lr1.predict(test_data)
>>> sum(lr1_predict_labels == test_labels)
446  


>>> lr2 = LogisticRegression()
>>> lr2.fit(new_train_data, training_labels)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,  
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0)
>>> lr2_predict_labels = lr2.predict(new_test_data)
>>> sum(lr2_predict_labels == test_labels)  # 正确率略微提升
449  
