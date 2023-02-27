from sklearn import svm
import pandas as pd
import numpy as np

df = pd.read_csv('//ad.uillinois.edu/engr-ews/chenyim2/Desktop/hw5_treasury yield curve data.csv')
df.columns
df = df.drop('Date',axis=1)

#split dataset
from sklearn.model_selection import train_test_split
df.columns
x = df.iloc[:,:-1].values
y = df['Adj_Close'].values
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state = 42)


#Normalize data 
from sklearn.preprocessing import StandardScaler
scalerx = StandardScaler().fit(x_train)
y_train = y_train.reshape(len(y_train),1)
y_train.shape
y_test = y_test.reshape(len(y_test),1)
y_test.shape
scalery = StandardScaler().fit(y_train)

x_train = scalerx.transform(x_train)
y_train = scalery.transform(y_train)
x_test = scalerx.transform(x_test)
y_test = scalery.transform(y_test)

#EDA
#heat map
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('correlation')
plt.show()

#create scatter plot matrix
cols = ['SVENF01', "SVENF05", 'SVENF15', 'SVENF30', 'Adj_Close']
sns.pairplot(df[cols],size=2.5)
plt.tight_layout()
plt.show()


#create correlation matrix
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm =sns.heatmap(cm,
                annot = True,
                square = True,
                fmt = '.2f',
                annot_kws = {'size':15},
                yticklabels = cols,
                xticklabels = cols)
plt.show()

    
#create method to evaluate training dataset
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
def train_and_evaluate(clf, x_train, y_train):    
    clf.fit(x_train, y_train)   
    print ("Coefficient of determination on training set:",clf.score(x_train, y_train))
    y_pred = clf.predict(x_train)
    mse = mean_squared_error(y_train,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE value ', rmse)
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, x_train, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
    y_pred = cross_val_predict(clf, x_train, y_train, cv=cv)
    mse = mean_squared_error(y_train,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE of 5-fold crossvalidaton ',rmse)
    return np.mean(scores), rmse

#linear regression
from sklearn import linear_model
clf_sgd=linear_model.SGDRegressor(loss='squared_error',random_state=42)
train_and_evaluate(clf_sgd,x_train,y_train)

#SVR
clf_svr =svm.SVR(kernel='linear')
clf_svr.fit(x_train, y_train)
train_and_evaluate(clf_svr,x_train,y_train)

##create method to evaluate training dataset
def test_and_evaluate(clf, x_test, y_test):    
    clf.fit(x_test, y_test)   
    print ("Coefficient of determination on test set:",clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE value ', rmse)
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, x_test, y_test, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
    y_pred = cross_val_predict(clf, x_test, y_test, cv=cv)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE of 5-fold crossvalidaton ',rmse)
    return np.mean(scores), rmse


#linear regression
clf_sgd=linear_model.SGDRegressor(loss='squared_error',random_state=42)
test_and_evaluate(clf_sgd,x_test,y_test)

#SVR
clf_svr =svm.SVR(kernel='linear')
clf_svr.fit(x_test, y_test)
test_and_evaluate(clf_svr,x_test,y_test)

#PCA Transfor
cov_matrix=np.cov(x_train.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_matrix)
print('Eigenvalues \n%s' % eigen_vals)

#calculates explained variance and cumulative explained variance
total=sum(eigen_vals)
variance_explained=np.array([(i/total) for i in sorted(eigen_vals,reverse=True)])
cum_variance_explained=np.cumsum(variance_explained)
print(cum_variance_explained)

#feature transformation using first 3 principal components
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis],
               eigen_pairs[2][1][:, np.newaxis]))

print('Matrix W:\n', w)
#transform both training and testing dataset
x_train_pca = x_train.dot(w)
x_test_pca=x_test.dot(w)

#create method to evaluate training dataset(PCA)
def train_and_evaluate_pca(clf, x_train_pca, y_train):    
    clf.fit(x_train_pca, y_train)   
    print ("Coefficient of determination on training set:",clf.score(x_train_pca, y_train))
    y_pred = clf.predict(x_train_pca)
    mse = mean_squared_error(y_train,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE value ', rmse)
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, x_train_pca, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
    y_pred = cross_val_predict(clf, x_train_pca, y_train, cv=cv)
    mse = mean_squared_error(y_train,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE of 5-fold crossvalidaton ',rmse)
    return np.mean(scores), rmse

#linear regression for training data
train_and_evaluate_pca(clf_sgd,x_train_pca,y_train)


#SVR for training data
clf_svr.fit(x_train_pca, y_train)
train_and_evaluate_pca(clf_svr,x_train_pca,y_train)

##create method to evaluate training dataset
def test_and_evaluate_pca(clf, x_test_pca, y_test):    
    clf.fit(x_test_pca, y_test)   
    print ("Coefficient of determination on test set:",clf.score(x_test_pca, y_test))
    y_pred = clf.predict(x_test_pca)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE value ', rmse)
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, x_test_pca, y_test, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
    y_pred = cross_val_predict(clf,x_test_pca, y_test, cv=cv)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('RMSE of 5-fold crossvalidaton ',rmse)
    return np.mean(scores), rmse

#linear regression for test data
test_and_evaluate_pca(clf_sgd,x_test_pca,y_test)


#SVR for test data
clf_svr.fit(x_test_pca, y_test)
test_and_evaluate_pca(clf_svr,x_test_pca,y_test)

print("My name is {Chenyi Mao}")
print("My NetID is: {chenyim2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")







