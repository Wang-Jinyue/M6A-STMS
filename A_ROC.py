from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import pandas as pd
#Arabidopsis thaliana dataset
A_datafile = u'F:\\N6-methyladenosine sites\\m6A\\Fusion\\LassoCV\\Arabidopsis thaliana.csv'
A_dataset = pd.read_csv(A_datafile,header = None)
X1 = A_dataset.values[:, 0:514]
y1 = A_dataset.values[:, 515]

A_clf1 = KNeighborsClassifier(n_neighbors=36, p=1, weights='distance', algorithm='auto', leaf_size=30, n_jobs=1)
A_clf2 = RandomForestClassifier(n_estimators=181, max_depth=13, max_features=21, min_samples_leaf=3, min_samples_split=10, criterion='gini', random_state=90)
A_clf3 = ExtraTreesClassifier(criterion='gini', max_depth=94, min_samples_split=14)
A_clf4 = BaggingClassifier(n_estimators=191, bootstrap=True)
A_clf5 = AdaBoostClassifier(n_estimators=96, learning_rate=0.3,  algorithm='SAMME')
A_lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, class_weight='balanced', solver='liblinear', C=0.1, max_iter=50, multi_class='ovr')
sclf1 = StackingCVClassifier( classifiers = [A_clf1, A_clf2, A_clf3, A_clf4, A_clf5],
                             use_probas = True,
                             meta_classifier = A_lr,
                             random_state=42)

def cv_roc(X,y,sclf):
    cross_score = []
    y_new = []
    cv = 10
    k_fold = KFold(n_splits=cv)
    for train,test in k_fold.split(X,y):
        sclf.fit(X[train], y[train])
        y_scores = sclf.predict_proba(X[test])[:,1]
        cross_score.extend(list(y_scores))
        y_new.extend(y[test])
    y_new = np.array(y_new)
    cross_score = np.array(cross_score)
    return y_new,cross_score


y_new, cross_score = cv_roc(X1,y1,sclf=A_clf1)
fpr1, tpr1, thresholds1 = roc_curve(y_new, cross_score)
roc_auc1 = auc(fpr1, tpr1)
y_new, cross_score = cv_roc(X1,y1,sclf=A_clf2)
fpr2, tpr2, thresholds2 = roc_curve(y_new, cross_score)
roc_auc2 = auc(fpr2, tpr2)
y_new, cross_score = cv_roc(X1,y1,sclf=A_clf3)
fpr3, tpr3, thresholds3 = roc_curve(y_new, cross_score)
roc_auc3 = auc(fpr3, tpr3)
y_new, cross_score = cv_roc(X1,y1,sclf=A_clf4)
fpr4, tpr4, thresholds4 = roc_curve(y_new, cross_score)
roc_auc4 = auc(fpr4, tpr4)
y_new, cross_score = cv_roc(X1,y1,sclf=A_clf5)
fpr5, tpr5, thresholds5 = roc_curve(y_new, cross_score)
roc_auc5 = auc(fpr5, tpr5)
y_new, cross_score = cv_roc(X1,y1,sclf=sclf1)
fpr6, tpr6, thresholds6 = roc_curve(y_new, cross_score)
roc_auc6 = auc(fpr6, tpr6)

p1 = plt.plot(fpr1, tpr1, linewidth = 2)
p2 = plt.plot(fpr2, tpr2, linewidth = 2)
p3 = plt.plot(fpr3, tpr3, linewidth = 2)
p4 = plt.plot(fpr4, tpr4, linewidth = 2)
p5 = plt.plot(fpr5, tpr5, linewidth = 2)
p6 = plt.plot(fpr6, tpr6, linewidth = 2)
plt.legend([p1, p2, p3, p4, p5, p6], labels=['KNN (area = %0.3f)' % roc_auc1,'RF (area = %0.3f)' % roc_auc2, 'ET (area = %0.3f)' % roc_auc3, 'Bagging (area = %0.3f)' % roc_auc4, 'Adaboost (area = %0.3f)' % roc_auc5, 'Stacking (area = %0.3f)' % roc_auc6] ,loc='lower right')
plt.plot([0,1], [0,1], 'k--')
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()