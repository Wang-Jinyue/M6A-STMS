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
#D.melanogaster dataset
D_datafile = u'F:\\N6-methyladenosine sites\\m6A\\Fusion\\LassoCV\\D.melanogaster\\10-3.csv'
D_dataset = pd.read_csv(D_datafile,header = None)
X2 = D_dataset.values[:, 0:278]
y2 = D_dataset.values[:, 279]
#Saccgaromyces cerevisiae
S_datafile = u'F:\\N6-methyladenosine sites\\m6A\\Fusion\\LassoCV\\Saccgaromyces cerevisiae.csv'
S_dataset = pd.read_csv(S_datafile,header = None)
X3 = S_dataset.values[:, 0:279]
y3 = S_dataset.values[:, 280]
#Human
H_datafile = u'F:\\N6-methyladenosine sites\\m6A\\Fusion\\LassoCV\\Human.csv'
H_dataset = pd.read_csv(H_datafile,header = None)
X4 = H_dataset.values[:, 0:300]
y4 = H_dataset.values[:, 301]

#Arabidopsis thaliana basic model
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

#D.melanogaster basic model
D_clf1 = KNeighborsClassifier(n_neighbors=36, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=1)
D_clf2 = RandomForestClassifier(n_estimators=161, max_depth=5, max_features=25, min_samples_leaf=9, min_samples_split=5, criterion='entropy', random_state=90)
D_clf3 = ExtraTreesClassifier(criterion='gini', max_depth=99, min_samples_split=14)
D_clf4 = BaggingClassifier(n_estimators=101, bootstrap=True)
D_clf5 = AdaBoostClassifier(n_estimators=78, learning_rate=0.2,  algorithm='SAMME')
D_lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, class_weight='balanced', solver='liblinear', C=0.2, max_iter=200, multi_class='ovr')
sclf2 = StackingCVClassifier( classifiers = [D_clf1, D_clf2, D_clf3, D_clf4, D_clf5],
                             use_probas = True,
                             meta_classifier = D_lr,
                             random_state=42)

#Saccgaromyces cerevisiae basic model
S_clf1 = KNeighborsClassifier(n_neighbors=28, p=1, weights='distance', algorithm='auto', leaf_size=30, n_jobs=1)
S_clf2 = RandomForestClassifier(n_estimators=121, max_depth=17, max_features=16, min_samples_leaf=1, min_samples_split=2, criterion='gini', random_state=90)
S_clf3 = ExtraTreesClassifier(criterion='gini', max_depth=27, min_samples_split=18)
S_clf4 = BaggingClassifier(n_estimators=101, bootstrap=True)
S_clf5 = AdaBoostClassifier(n_estimators=36, learning_rate=0.7,  algorithm='SAMME')
S_lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, class_weight='balanced', solver='liblinear', C=0.6, max_iter=100, multi_class='ovr')
sclf3 = StackingCVClassifier( classifiers = [S_clf1, S_clf2, S_clf3, S_clf4, S_clf5],
                             use_probas = True,
                             meta_classifier = S_lr,
                             random_state=42)

#Huamn basic model
H_clf1 = KNeighborsClassifier(n_neighbors=37, p=1, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=1)
H_clf2 = RandomForestClassifier(n_estimators=141, max_depth=5, max_features=26, min_samples_leaf=14, min_samples_split=19, criterion='entropy', random_state=90)
H_clf3 = ExtraTreesClassifier(criterion='entropy', max_depth=5, min_samples_split=13)
H_clf4 = BaggingClassifier(n_estimators=191, bootstrap=True)
H_clf5 = AdaBoostClassifier(n_estimators=11, learning_rate=0.7,  algorithm='SAMME')
H_lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, class_weight='balanced', solver='liblinear', C=0.2, max_iter=150, multi_class='ovr')
sclf4 = StackingCVClassifier( classifiers = [H_clf1, H_clf2, H_clf3, H_clf4, H_clf5],
                             use_probas = True,
                             meta_classifier = H_lr,
                             random_state=42)

print('10-fold cross validation:\n')
y_train_pred1 = model_selection.cross_val_predict(sclf1, X1, y1, cv=10)
print(confusion_matrix(y1, y_train_pred1))
y_train_pred2 = model_selection.cross_val_predict(sclf2, X2, y2, cv=10)
print(confusion_matrix(y2, y_train_pred2))
y_train_pred3 = model_selection.cross_val_predict(sclf3, X3, y3, cv=10)
print(confusion_matrix(y3, y_train_pred3))
y_train_pred4 = model_selection.cross_val_predict(sclf4, X4, y4, cv=10)
print(confusion_matrix(y4, y_train_pred4))

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


y_new, cross_score = cv_roc(X1,y1,sclf=sclf1)
fpr1, tpr1, thresholds1 = roc_curve(y_new, cross_score)
roc_auc1 = auc(fpr1, tpr1)
y_new, cross_score = cv_roc(X2,y2,sclf=sclf2)
fpr2, tpr2, thresholds2 = roc_curve(y_new, cross_score)
roc_auc2 = auc(fpr2, tpr2)
y_new, cross_score = cv_roc(X3,y3,sclf=sclf3)
fpr3, tpr3, thresholds3 = roc_curve(y_new, cross_score)
roc_auc3 = auc(fpr3, tpr3)
y_new, cross_score = cv_roc(X4,y4,sclf=sclf4)
fpr4, tpr4, thresholds4 = roc_curve(y_new, cross_score)
roc_auc4 = auc(fpr4, tpr4)


p1 = plt.plot(fpr1, tpr1, linewidth = 2)
p2 = plt.plot(fpr2, tpr2, linewidth = 2)
p3 = plt.plot(fpr3, tpr3, linewidth = 2)
p4 = plt.plot(fpr4, tpr4, linewidth = 2)
plt.legend([p1, p2, p3, p4], labels=['Arabidopsis thaliana (area = %0.3f)' % roc_auc1,'D.melanogaster (area = %0.3f)' % roc_auc2, 'Saccgaromyces cerevisiae (area = %0.3f)' % roc_auc3, 'Human (area = %0.3f)' % roc_auc4] ,loc='lower right')
plt.plot([0,1], [0,1], 'k--')
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()