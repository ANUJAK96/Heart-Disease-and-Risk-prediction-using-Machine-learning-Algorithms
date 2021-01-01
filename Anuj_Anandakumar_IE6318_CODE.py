#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
data = pd.read_csv("framingham.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


data.drop_duplicates()
data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.fillna(0, inplace = True)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# In[ ]:


convert_dict = {"male" : str,
               "education" : str,
               "currentSmoker" : str,
               "BPMeds" : str,
               "prevalentStroke" : str,
               "prevalentHyp" : str,
               "diabetes" : str,
               "TenYearCHD" : str}


# In[ ]:


data = data.astype(convert_dict)
data.dtypes


# In[ ]:


list(set(data.dtypes.tolist()))


# In[ ]:


plt.figure(figsize=(20,10), facecolor='w')
sb.boxplot(data=data)
plt.show()


# In[ ]:


print("the max totChol is", data["totChol"].max())
print("the max sysBP is", data["sysBP"].max())
print("the max diaBP is", data["diaBP"].max())
print("the max BMI is", data["BMI"].max())
print("the max heartRate is", data["heartRate"].max())
print("the max glucose is", data["glucose"].max())


# In[ ]:


data = data[data["totChol"] < 522]
data = data[data["sysBP"] < 221]
data = data[data["diaBP"] < 107]
data = data[data["BMI"] < 43]
data = data[data["heartRate"] < 107]
data = data[data["glucose"] < 296]


# In[ ]:


data.shape


# In[ ]:


data_num = data.select_dtypes(include = ["float64", "int64"])
data_num.head()


# In[ ]:


data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[ ]:


numeric_features = ['cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
for feature in numeric_features:
    plt.figure(figsize=(18, 10), facecolor='w')
    sb.distplot(data[feature])
    plt.title('{} Distribution'.format(feature), fontsize=20)
    plt.show()


# In[ ]:


sb.heatmap(data_num.corr(), annot = True, cmap = "magma")


# In[ ]:


categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD']
for feature in categorical_features:
    plt.figure(figsize=(18, 10), facecolor='w')
    sb.countplot(data[feature])
    plt.title('{} Distribution'.format(feature), fontsize=20)
    plt.show()


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


x = data.iloc[:, 0:14]
y = data.iloc[:, 15]


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.feature_selection import SelectKBest
topfeatures = SelectKBest(score_func = chi2, k = 10)
fit = topfeatures.fit(x, y)
xyscore = pd.DataFrame(fit.scores_)
xycolumn = pd.DataFrame(x.columns)


# In[ ]:


featurescore = pd.concat([xycolumn, xyscore], axis = 1)
featurescore.columns = ["Feature", "Score"]


# In[ ]:


featurescore = featurescore.sort_values(by = "Score", ascending= False)
featurescore


# In[ ]:


plt.figure(figsize= (20,5))
sb.barplot(x = "Feature", y = "Score", data = featurescore)
plt.box(False)
plt.title("Features Ranked by Scores", fontsize = 15)
plt.xlabel("\n Feature", fontsize = 13)
plt.ylabel("Importance \n", fontsize = 13)
plt.xticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.show()


# In[ ]:


features_list = featurescore["Feature"].tolist()[:10]
features_list


# In[ ]:


data = data[["male", "age", "cigsPerDay", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "TenYearCHD"]]
data.head()


# In[ ]:


sb.heatmap(data.corr(), annot = True, cmap = "magma")


# In[ ]:


from sklearn.preprocessing import MinMaxScaler 


# In[ ]:


scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform (data), columns = data.columns)
data_scaled.describe()


# In[ ]:


x = data_scaled.drop(["TenYearCHD"], axis = 1)
y = data_scaled["TenYearCHD"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
len(X_train)
len(X_test)


# In[ ]:


import six
import sys
sys.modules['sklearn.externals.six'] = six
#import mlrose
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state= 2)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state= 2)


# In[ ]:


X_train, y_train = smote.fit_resample(X_train, y_train)
len(X_train)
len(y_train)


# In[ ]:


import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report


# In[ ]:


from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression().fit(X_train, y_train)
model_lr_pred = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, model_lr_pred)
print(f"The Accuracy score for Logistic Regression Model is: {round(acc_lr,3)*100}%")
f1_lr = f1_score(y_test, model_lr_pred)
print(f"The f1 score for Logistic Regression Model is: {round(f1_lr, 3)*100}%")
prec_lr = precision_score(y_test, model_lr_pred)
print(f"The precision for Logistic Regression Model is: {round(prec_lr, 3)*100}%")
recall_lr = recall_score(y_test, model_lr_pred)
print(f"The sensitivity for Logistic Regression Model is: {round(recall_lr, 3)*100}%")
confmat_lr = confusion_matrix(y_test, model_lr_pred)
print(f"Confusion Matrix:", "\n", confmat_lr)
conf_matrix_lr = pd.DataFrame(data=confmat_lr,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_lr), annot = True, cmap = "magma", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_lr_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Logistic Regression')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_lr = roc_auc_score(y_test, model_lr_pred)
print(f"The Area Under Curve is {round(auc_lr, 3)*100}%")


# In[ ]:


from sklearn.svm import SVC
model_svm = SVC().fit(X_train, y_train)
model_svm_pred = model_svm.predict(X_test)
acc_svm = accuracy_score(y_test, model_svm_pred)
print(f"The Accuracy score for SVM Model is: {round(acc_svm,3)*100}%")
f1_svm = f1_score(y_test, model_svm_pred)
print(f"The f1 score for SVM Model is: {round(f1_svm, 3)*100}%")
prec_svm = precision_score(y_test, model_svm_pred)
print(f"The precision for SVM Model is: {round(prec_svm, 3)*100}%")
recall_svm = recall_score(y_test, model_svm_pred)
print(f"The sensitivity for SVM Model is: {round(recall_svm, 3)*100}%")
confmat_svm = confusion_matrix(y_test, model_svm_pred)
print(f"Confusion Matrix:", "\n", confmat_svm)
conf_matrix_svm = pd.DataFrame(data=confmat_svm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_svm), annot = True, cmap = "viridis", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_svm_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for SVM')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_svm = roc_auc_score(y_test, model_svm_pred)
print(f"The Area Under Curve is {round(auc_svm, 3)*100}%")


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(min_samples_split= 50, random_state= 2).fit(X_train, y_train)
model_dt_pred = model_dt.predict(X_test)
acc_dt = accuracy_score(y_test, model_dt_pred)
print(f"The Accuracy score for Decision Tree Model is: {round(acc_dt,3)*100}%")
f1_dt = f1_score(y_test, model_dt_pred)
print(f"The f1 score for Decision Tree Model is: {round(f1_dt, 3)*100}%")
prec_dt = precision_score(y_test, model_dt_pred)
print(f"The precision for Decision Tree Model is: {round(prec_dt, 3)*100}%")
recall_dt = recall_score(y_test, model_dt_pred)
print(f"The sensitivity for Logistic Regression Model is: {round(recall_dt, 3)*100}%")
confmat_dt = confusion_matrix(y_test, model_dt_pred)
print(f"Confusion Matrix:", "\n", confmat_dt)
conf_matrix_dt = pd.DataFrame(data=confmat_dt,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_dt_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Decision Tree')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_dt = roc_auc_score(y_test, model_dt_pred)
print(f"The Area Under Curve is {round(auc_dt, 3)*100}%")


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors = 10).fit(X_train, y_train)
model_knn_pred = model_knn.predict(X_test)
acc_knn = accuracy_score(y_test, model_knn_pred)
print(f"The Accuracy score for KNN Model is: {round(acc_knn,3)*100}%")
f1_knn = f1_score(y_test, model_knn_pred)
print(f"The f1 score for KNN Model is: {round(f1_knn, 3)*100}%")
prec_knn = precision_score(y_test, model_knn_pred)
print(f"The precision for KNN Model is: {round(prec_knn, 3)*100}%")
recall_knn = recall_score(y_test, model_knn_pred)
print(f"The sensitivity for KNN Model is: {round(recall_knn, 3)*100}%")
confmat_knn = confusion_matrix(y_test, model_knn_pred)
print(f"Confusion Matrix:", "\n", confmat_knn)
conf_matrix_knn = pd.DataFrame(data=confmat_knn,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_knn), annot = True, cmap = "Spectral", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_knn_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for KNN')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_knn = roc_auc_score(y_test, model_knn_pred)
print(f"The Area Under Curve is {round(auc_knn, 3)*100}%")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators = 100, random_state = 55, max_depth = 12).fit(X_train, y_train)
model_rf_pred = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, model_rf_pred)
print(f"The Accuracy score for Random Forest Model is: {round(acc_rf,3)*100}%")
f1_rf = f1_score(y_test, model_rf_pred)
print(f"The f1 score for Random Forest Model is: {round(f1_rf, 3)*100}%")
prec_rf = precision_score(y_test, model_rf_pred)
print(f"The precision for Random Forest Model is: {round(prec_rf, 3)*100}%")
recall_rf = recall_score(y_test, model_rf_pred)
print(f"The sensitivity for Random Forest Model is: {round(recall_rf, 3)*100}%")
confmat_rf = confusion_matrix(y_test, model_rf_pred)
print(f"Confusion Matrix:", "\n", confmat_rf)
conf_matrix_rf = pd.DataFrame(data=confmat_rf,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_rf), annot = True, cmap = "icefire", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_rf_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Random Forest')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_rf = roc_auc_score(y_test, model_rf_pred)
print(f"The Area Under Curve is {round(auc_rf, 3)*100}%")


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model_gb = GradientBoostingClassifier(random_state = 55).fit(X_train, y_train)
model_gb_pred = model_gb.predict(X_test)
acc_gb = accuracy_score(y_test, model_gb_pred)
print(f"The Accuracy score for Gradient Boosting Model is: {round(acc_gb,3)*100}%")
f1_gb = f1_score(y_test, model_gb_pred)
print(f"The f1 score for Gradient Boosting Model is: {round(f1_gb, 3)*100}%")
prec_gb = precision_score(y_test, model_gb_pred)
print(f"The precision for Gradient Boosting Model is: {round(prec_gb, 3)*100}%")
recall_gb = recall_score(y_test, model_gb_pred)
print(f"The sensitivity for Gradient Boosting Model is: {round(recall_gb, 3)*100}%")
confmat_gb = confusion_matrix(y_test, model_gb_pred)
print(f"Confusion Matrix:", "\n", confmat_gb)
conf_matrix_gb = pd.DataFrame(data=confmat_gb,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_gb), annot = True, cmap = "magma_r", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_gb_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Gradient Boosting')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_gb = roc_auc_score(y_test, model_gb_pred)
print(f"The Area Under Curve is {round(auc_gb, 3)*100}%")


# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


# Creating RF model
rf = RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator= rf, param_distributions= random_grid, n_iter= 150, cv = 5, verbose= 3, 
                               random_state= 10, n_jobs= -1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_rf = rf_random.fit(X_train, y_train)
model_rf_pred = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, model_rf_pred)
print(f"The Accuracy score for Random Forest Model is: {round(acc_rf,3)*100}%")
f1_rf = f1_score(y_test, model_rf_pred)
print(f"The f1 score for Random Forest Model is: {round(f1_rf, 3)*100}%")
prec_rf = precision_score(y_test, model_rf_pred)
print(f"The precision for Random Forest Model is: {round(prec_rf, 3)*100}%")
recall_rf = recall_score(y_test, model_rf_pred)
print(f"The sensitivity for Random Forest Model is: {round(recall_rf, 3)*100}%")
confmat_rf = confusion_matrix(y_test, model_rf_pred)
print(f"Confusion Matrix:", "\n", confmat_rf)
conf_matrix_rf = pd.DataFrame(data=confmat_rf,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_rf), annot = True, cmap = "icefire", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_rf_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Random Forest')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_rf = roc_auc_score(y_test, model_rf_pred)
print(f"The Area Under Curve is {round(auc_rf, 3)*100}%")


# In[ ]:


n_estimators = [int(i) for i in np.linspace(start=100,stop=1000,num=10)]
max_features = ['auto','sqrt']
max_depth = [int(i) for i in np.linspace(10, 100, num=10)]
max_depth.append(None)
min_samples_split=[2,5,10]
min_samples_leaf = [1,2,4]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[ ]:


gb=GradientBoostingClassifier(random_state=0)
gb_random = RandomizedSearchCV(estimator=gb, param_distributions=random_grid,
                              n_iter=75, scoring='f1', 
                              cv=2, verbose=2, random_state=0, n_jobs=-1,
                              return_train_score=True)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model_gb = gb_random.fit(X_train, y_train)
model_gb_pred = model_gb.predict(X_test)
acc_gb = accuracy_score(y_test, model_gb_pred)
print(f"The Accuracy score for Gradient Boosting Model is: {round(acc_gb,3)*100}%")
f1_gb = f1_score(y_test, model_gb_pred)
print(f"The f1 score for Gradient Boosting Model is: {round(f1_gb, 3)*100}%")
prec_gb = precision_score(y_test, model_gb_pred)
print(f"The precision for Gradient Boosting Model is: {round(prec_gb, 3)*100}%")
recall_gb = recall_score(y_test, model_gb_pred)
print(f"The sensitivity for Gradient Boosting Model is: {round(recall_gb, 3)*100}%")
confmat_gb = confusion_matrix(y_test, model_gb_pred)
print(f"Confusion Matrix:", "\n", confmat_gb)
conf_matrix_gb = pd.DataFrame(data=confmat_gb,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sb.heatmap(pd.DataFrame(conf_matrix_gb), annot = True, cmap = "magma_r", fmt = "g")


# In[ ]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model_gb_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Gradient Boosting')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Area Under Curve (AUC)
auc_gb = roc_auc_score(y_test, model_gb_pred)
print(f"The Area Under Curve is {round(auc_gb, 3)*100}%")


# In[ ]:


FPR_lr, TPR_lr, threshold_lr = roc_curve(y_test, model_lr_pred)
FPR_svm, TPR_svm, threshold_svm = roc_curve(y_test, model_svm_pred)
FPR_dt, TPR_dt, threshold_dt = roc_curve(y_test, model_dt_pred)
FPR_knn, TPR_knn, threshold_knn = roc_curve(y_test, model_knn_pred)
FPR_rf, TPR_rf, threshold_rf = roc_curve(y_test, model_rf_pred)
FPR_gb, TPR_gb, threshold_gb = roc_curve(y_test, model_gb_pred) 
sb.set_style("whitegrid")
plt.figure(figsize = (15,8), facecolor = "w")
plt.title("Receiver Operating Characteristics Curve")
plt.plot(FPR_lr, TPR_lr, label = "Logistic Regression")
plt.plot(FPR_svm, TPR_svm, label = "Support Vector Machine")
plt.plot(FPR_dt, TPR_dt, label = "Decision Tree")
plt.plot(FPR_knn, TPR_knn, label = "KNearest Neighbor")
plt.plot(FPR_rf, TPR_rf, label = "Random Forest")
plt.plot(FPR_gb, TPR_gb, label = "Gradient Booster")
plt.plot([0,1], ls = "--")
plt.plot([0,0], [1,0], c = "0.5")
plt.plot([1,1], c = "0.5")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend()
plt.show()


# In[ ]:




