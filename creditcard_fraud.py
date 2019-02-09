##Dataset downloaded from https://www.kaggle.com/mlg-ulb/creditcardfraudcreditcard.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from os import chdir
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
chdir('c:\\users\\jeryl\\desktop\\Python\\kaggle\\creditcardfraud')

#Reading data
train = pd.read_csv('creditcard.csv', header=0)

#EDA
train.hist(bins=15, edgecolor='black')
plt.show()

sns.set(font_scale=0.5)
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.show()

#Split data into features and labels
X,y=train.iloc[:,0:30], train.iloc[:,30]

#Scale features
scaler = StandardScaler() 
rescaledX = scaler.fit_transform(X)

#PCA
pca = PCA()
pca.fit(rescaledX)
plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
plt.xlabel('Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance of Each Component')
plt.show()

#N components accounting for 90% of variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_var)
plt.plot([0,29],[0.9,0.9], linestyle='-')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()
n_components = 25

#Final fit of PCA with N components
pca = PCA(n_components, random_state=10)
pca_X = pca.fit_transform(rescaledX)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(pca_X, y, random_state=10)

#Logistic Regression model
lr = LogisticRegression(random_state=10, class_weight='balanced', solver='lbfgs')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
class_rep = classification_report(y_test, y_pred)
con_matrix = confusion_matrix(y_test, y_pred)
print("Classification Report Unbal: \n", class_rep)
print('\n')
print("Confusion Matrix Unbal: \n", con_matrix)
print('\n'*3)

#Balancing the classes 0:492, 1:492
true_transact = train[train.Class == 0]
fraud_transact = train[train.Class == 1]
sampled_true_transact = true_transact.sample(n=492)
total = pd.concat([sampled_true_transact, fraud_transact])

features = total.drop('Class', axis=1)
labels = total.Class
pca_features = pca.fit_transform(scaler.fit_transform(features))

X_train2, X_test2, y_train2, y_test2 = train_test_split(pca_features, labels, random_state=10)
lr2 = LogisticRegression(random_state=10, solver='lbfgs')
lr2.fit(X_train2, y_train2)
y_pred2 = lr2.predict(X_test2)
class_rep_bal = classification_report(y_test2, y_pred2)
con_matrix_bal = confusion_matrix(y_test2, y_pred2)
print("Classification Report Bal: \n", class_rep_bal)
print('\n')
print("Confusion Matrix Bal: \n", con_matrix_bal)
print('\n'*3)

#Cross-Validation
cv_results = cross_val_score(lr2, pca_features, labels, cv=10)
print('Cross Validation Score LogReg: ',np.mean(cv_results))
print('\n'*3)

#Trying other models: NaiveBayes
gnb = GaussianNB()
gnb.fit(X_train2, y_train2)
y_pred3 = gnb.predict(X_test2)
class_rep_bal_gnb = classification_report(y_test2, y_pred3)
con_matrix_bal_gnb = confusion_matrix(y_test2, y_pred3)
print("Classification Report GNB: \n", class_rep_bal_gnb)
print('\n')
print("Confusion Matrix GNB: \n", con_matrix_bal_gnb)
print('\n'*3)

#Trying other models: KNeighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train2, y_train2)
y_pred4 = knn.predict(X_test2)
class_rep_bal_knn = classification_report(y_test2, y_pred4)
con_matrix_bal_knn = confusion_matrix(y_test2, y_pred4)
print("Classification Report KNN: \n", class_rep_bal_knn)
print('\n')
print("Confusion Matrix KNN: \n", con_matrix_bal_knn)
print('\n'*3)

#GridSearch CV of KNeighbors
param_grid = {'n_neighbors': np.arange(1,20)}
knn_cv = GridSearchCV(knn, param_grid, cv=10)
knn_cv.fit(X_train2, y_train2)
print('Best n_neighbors: ', knn_cv.best_params_)
print('Best CV score: ', knn_cv.best_score_)
