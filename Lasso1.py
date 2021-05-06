import pandas as pd
from sklearn.linear_model import Lasso
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

datafile = 'F:\\N6-methyladenosine sites\\m6A\\Fusion\\csv\\Saccgaromyces cerevisiae.csv'
data = pd.read_csv(datafile, header = None)
scaler = StandardScaler()
X = data.values[:, 0:509]
Y = data.values[:, 510]

model_lasso = LassoCV(alphas=[0.1,1,0.001,0.0005]).fit(X,Y)
print(model_lasso.alpha_)
print(model_lasso.coef_)
X_selected = X[:,model_lasso.coef_!=0]

pd.DataFrame(X_selected).to_csv('F:\\N6-methyladenosine sites\\m6A\\Fusion\\LassoCV\\Saccgaromyces cerevisiae.csv', header = False, index = False)

coef = pd.Series(model_lasso.coef_)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
