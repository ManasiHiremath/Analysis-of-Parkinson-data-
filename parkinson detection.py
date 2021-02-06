#!/usr/bin/env python
# coding: utf-8

# Title: Parkinsons Disease Data Set
# 
# Abstract: Oxford Parkinson's Disease Detection Dataset
# 
# Source:
# 
# The dataset was created by Max Little of the University of Oxford, in 
# collaboration with the National Centre for Voice and Speech, Denver, 
# Colorado, who recorded the speech signals. The original study published the 
# feature extraction methods for general voice disorders.
# 
# 
# Attribute Information:
# 
# 1. Matrix column entries (attributes):
# 2. name - ASCII subject name and recording number
# 3. MDVP:Fo(Hz) - Average vocal fundamental frequency
# 4. MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
# 5. MDVP:Flo(Hz) - Minimum vocal fundamental frequency
# 6. MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several 
# 7. measures of variation in fundamental frequency
# 8. MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
# 9. NHR,HNR - Two measures of ratio of noise to tonal components in the voice
# 10. status - Health status of the subject (one) - Parkinson's, (zero) - healthy
# 11. RPDE,D2 - Two nonlinear dynamical complexity measures
# 12. DFA - Signal fractal scaling exponent
# 13. spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation 
# 
# Objective:
# Goal is to classify the patients into the respective labels using the attributes from
# their voice recordings

# In[1]:


#import required libraries


# In[88]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew 
import seaborn as sns
import scipy.stats
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[134]:


#read data
data=pd.read_csv('Data - Parkinsons.csv')


# ### Step 1: Understand data

# In[135]:


data.shape #(195, 24)
data.info()


# In[136]:


#checking for missing data
data.isnull().mean()*100


# Summary:
# 1. Data has records of 195 patients, across 23 voice recording measures. 
# 2. Name column has 195 unique records same as data records. So data has unique record per patient
# 2. Data has no missing observations.
# 3. Name of the patients wont be useful, so dropping it from further analysis
# 4. All other variables are either integer or float 64.
# 5. Converting Status as object for EDA

# ### Step 2: EDA

# In[137]:


#improving display format of describe method
pd.set_option('float_format', '{:f}'.format)
#converting status to object for EDA
data_sub= data.copy()
data_sub.sample(5)


# In[138]:


# descriptive statistics of each of the attributes
data_sub['status']= data_sub['status'].astype('object')
data_sub.describe(include ='all').transpose()


# Summary:
# 1. Average vocal fundamental frequency: MDVP:Fo(Hz) has average frequency 154.229 HZ
# 2. Maximum vocal fundamental frequency:MDVP:Fhi(Hz) has average frequency 197.105 HZ, max value looks quite deviated from 75% percenile. Data looks skewed and might have the presence of outliers
# 3. Minimum vocal fundamental frequency:MDVP:Flo(Hz) has average frequency of 116.325 Hz.
# 4. MDVP:Jitter(%): has a small percentage value, with average jitter(%) 0.00622046
# 5. MDVP:Jitter(Abs): has a small scale compared to other variables and has an average of 0.000044
# 6. Status has two values, 1 being most frequent, with frequency 147
# 7. spread1 : is on negative scale with average of -5.684397
# 8. Scale of each attributes is varying a lot. Data will need to be scaled
# 

# ### Step 3: Univariate Analysis

# In[139]:


if 'name' in data_sub.columns:
    data_sub=data_sub.drop(columns='name', axis=1)
    print('column name has been dropped')
else:
    print('Name of the patient does not exist')


# In[140]:


data_sub['status'].value_counts(normalize=True).plot(kind='bar')


# Summary
# 1. status - Health status of the subject (one) - Parkinson's, (zero) - healthy
# 2. data has 147(75%) patients identified for parkiensons and 48 (25%) patients not identified as Parkinson

# In[141]:


#Check distribution of each variable and also check the skwness of variables.
# Descriptive statistics shows presense of skwness
def skewness(data,x):
    val=data[x].skew(axis = 0,skipna = True)
    if val < -1.0 or val > 1.0:
        print('{} is highly skewed with skweness score : {}'.format(x,val))
    elif (val > -1 and val <-0.5) or (val < 1 and val >0.5):
        print('{} is moderatly skewed with skweness score : {}'.format(x,val))
    elif val > -0.5 or val < 0.5:
        print('{} is approximately symmetric with skweness score : {}'.format(x,val))
    return ''

count=1
for i in data_sub.columns:
    if (data_sub[i].dtypes == 'int64') or (data_sub[i].dtypes == 'float64'):
        print('Plot','-',count,':','Distribution Plot of',i)
        sns.histplot(data_sub[i],kde=True)
        plt.show()
        print(skewness(data_sub,i))
        count+=1


# Summary:
# 1. MDVP:Fo(Hz) is moderatly skewed with skweness score : 0.5917374636540784
# 2. MDVP:Fhi(Hz) is highly skewed with skweness score : 2.542145997588398
# 3. MDVP:Flo(Hz) is highly skewed with skweness score : 1.217350448627808
# 4. MDVP:Jitter(%) is highly skewed with skweness score : 3.0849462014441817
# 5. MDVP:Jitter(Abs) is highly skewed with skweness score : 2.6490714165257274
# 6. MDVP:RAP is highly skewed with skweness score : 3.360708450480554
# 7. MDVP:PPQ is highly skewed with skweness score : 3.073892457888517
# 8. Jitter:DDP is highly skewed with skweness score : 3.3620584478857203
# 9. MDVP:Shimmer is highly skewed with skweness score : 1.6664804101559663
# 10. MDVP:Shimmer(dB) is highly skewed with skweness score : 1.999388639086127
# 11. Shimmer:APQ3 is highly skewed with skweness score : 1.5805763798815677
# 12. Shimmer:APQ5 is highly skewed with skweness score : 1.798697066537622
# 13. MDVP:APQ is highly skewed with skweness score : 2.618046502215422
# 14. Shimmer:DDA is highly skewed with skweness score : 1.5806179936782263
# 15. NHR is highly skewed with skweness score : 4.22070912913906
# 16. HNR is moderatly skewed with skweness score : -0.5143174975652068
# 17. RPDE is approximately symmetric with skweness score : -0.14340241379821705
# 18. DFA is approximately symmetric with skweness score : -0.03321366071383484
# 19. spread1 is approximately symmetric with skweness score : 0.4321389320131796
# 20. spread2 is approximately symmetric with skweness score : 0.14443048549278412
# 21. D2 is approximately symmetric with skweness score : 0.4303838913329283
# 22. PPE is moderatly skewed with skweness score : 0.7974910716463578
# 23. Most of the variables are skewed. 

# In[142]:


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print('mean:{} and CI:({},{})'.format(m,m-h, m+h))
    return 

count=1
for i in data_sub.columns:
    if (data_sub[i].dtypes == 'int64') or (data_sub[i].dtypes == 'float64'):
        print('Plot','-',count,':',"Distribution of Parkinson's Disease vs",i)
        sns.boxplot(x='status',y=i, data=data_sub)
        plt.show()
        print('Group mean',(data_sub.groupby(by='status',)[i]).mean())
        print((data_sub.groupby(by='status',)[i]).apply(mean_confidence_interval))
        print('Group Median',(data_sub.groupby(by='status',)[i]).median())
        print('')
        print('**********************************************')
        print('')
        count+=1


# In[10]:


sns.pairplot(data_sub, hue='status',markers=["o", "s"], corner=True); # Get only lower triangle


# In[143]:


corr = abs(data_sub.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (20,8))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= "YlGnBu", annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
plt.xticks(rotation = 30)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# In[144]:


## Getting a list of attributes which are correlated
df=data_sub.copy()
df.drop(['status'],axis=1, inplace=True)
data_corr_val=pd.DataFrame()

def data_multicollinearity(data_for_corr):
    var1=[]
    var2=[]
    corr_var_list=[]
    data_corr_val=pd.DataFrame()
    for i in data_for_corr.columns:
        for j in data_for_corr.columns:
            if i== j:
                pass
            else:
                cor_var= data_for_corr[i].corr(data_for_corr[j],'pearson')
                var1.append(i)
                var2.append(j)
                corr_var_list.append(cor_var)
        
    data_corr_val=pd.DataFrame({'var1':var1,
                               'var2':var2,
                               'correlation':corr_var_list})
    data_corr_val=data_corr_val[(abs(data_corr_val['correlation'])> 0.85)]
    return data_corr_val
data_all=data_multicollinearity(df)
data_all.head()


# Summary:
# 1. Above is table of attributes which are correlated with each other. 
# 2. This needs to be taken care of else this will add redundancy in model

# ### Modelling 

# ## Step 1: Modelling with K-Fold Cross validation

# In[145]:


import sklearn.metrics
a=sklearn.metrics.SCORERS.keys()
a


# In[146]:


from sklearn import model_selection
from sklearn.metrics import auc
import time

list_of_models=[LogisticRegression,
               DecisionTreeClassifier,
               KNeighborsClassifier,
               GaussianNB,
               SVC,
               RandomForestClassifier]

def train_model_with_cv(model_list,X,y):
    for model in list_of_models:
        start = time.time()
        cls=model()
        kfold=model_selection.StratifiedKFold(n_splits=10, random_state=7)
        recall=model_selection.cross_val_score(cls,X,y,scoring='recall',cv=kfold)
        precision=model_selection.cross_val_score(cls,X,y,scoring='precision',cv=kfold)  
        auc=model_selection.cross_val_score(cls,X,y,scoring='roc_auc',cv=kfold)  
        end = time.time()
        duration = end - start  # calculate the total duration
        print(f"{model.__name__:22}", f" RECALL:{recall.mean():.3f} PRECISION:{precision.mean():.3f} AVGAUC:{auc.mean():.3f} AUCSTD:{auc.std():.2f} ELAPSED:{duration:.3f}" )
    return ""


# In[151]:


drop_list=['status','name']

if set(drop_list).issubset(data.columns):
    data_x=data.drop(drop_list, axis=1)
    print('Columns from drop list has been removed')
else:
    print("Columns in drop list does not exist")


# In[149]:


X_org=data_x
y_org=data['status']

data_iteration1=train_model_with_cv(list_of_models,X_org,y_org)


# ### Iteration 2
# 1. Since each variable is in different scale, in this iteration using MinMaxsclar to standardize all the numerical variables.
# 2. Dropping 'name' from all analysis.
# 3. In this iteration each algorithm will be analysed seperately. will be assembled together in later iterations
# 4. Will check the performance of MinMax scaler, considering skewness within attributes this might impact results

# In[155]:


#Rescaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_min_max=pd.DataFrame(scaler.fit_transform(data_x),index=data_x.index, columns=data_x.columns)
y_min_max=data['status']

data_iteration2=train_model_with_cv(list_of_models,X_min_max,y_min_max)


# ### Iteration 3

# In[158]:


from sklearn.preprocessing import StandardScaler
sca=StandardScaler()

X_std=pd.DataFrame(sca.fit_transform(data_x),index=data_x.index, columns=data_x.columns)
y_std=data['status']
data_iteration3=train_model_with_cv(list_of_models,X_std,y_std)


# ### Iteration 4

# In[162]:


fs = SelectKBest(score_func=f_classif,k=5)

# apply feature selection
X_selected = fs.fit(X_std, y_std)
cols = fs.get_support(indices=True)
features_df_new = X_std.iloc[:,cols]
var_imp=features_df_new.columns.to_list()

data_var_imp=X_std[var_imp]
data_var_imp_cor=data_multicollinearity(data_var_imp)
data_var_imp_cor

X_var_imp=data_var_imp
y_var_imp=data['status']
data_iteration4=train_model_with_cv(list_of_models,X_var_imp,y_var_imp)


# ### Stacking

# In[185]:


from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline

X_stack=pd.DataFrame(sca.fit_transform(data_x),index=data_x.index, columns=data_x.columns)
y_stack=data['status']
list_of_models=[
               ('DTree',DecisionTreeClassifier()),
               ('KNN',KNeighborsClassifier())]
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', SVC(random_state=42))]
clf = StackingClassifier(estimators=list_of_models, final_estimator=LogisticRegression())

X_train, X_test, y_train, y_test = train_test_split(X_stack, y_stack, test_size=0.3,stratify=y, random_state=42)
clf.fit(X_train, y_train).score(X_test, y_test)


# In[165]:


# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify = y)# Adding option stratify to get equal distributon of y in train and test
# print("Target data distribution in Train")
# print(y_train.value_counts(normalize=True))
# print("Target data distribution in Test")
# print(y_test.value_counts(normalize=True))


# In[14]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

def train_test_exp(model, X_train, X_test, y_train, y_test):    
    model.fit(X_train, y_train)   # fit the model with the train data
    y_pred = model.predict(X_test)  # make predictions on the test set
    score = round(model.score(X_test, y_test), 3)   # compute accuracy score for test set
    
    recall=metrics.recall_score(y_test, y_pred)
    precision=metrics.precision_score(y_test, y_pred)
    f1_score= round(metrics.f1_score(y_test, y_pred),3)
    
    probas_ = model.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = round(auc(fpr, tpr),3)
    
    return score, recall, precision,f1_score,roc_auc   # return all the metrics


# In[17]:


classfierdict={'model': [LogisticRegression(solver='liblinear'),
                KNeighborsClassifier(weights='distance'),
                GaussianNB(),
                svm.SVC(C=3, gamma=0.025,probability=True)]}

scoring_metric=pd.DataFrame()
model_name_list= []
score_list = []
recall_list=[]
precision_list=[]
f1_score_list=[]
roc_auc_list=[]

for i in classfierdict['model']:
    score,recall,precision,f1_score,roc_auc=train_test_exp(i,x_train, x_test, y_train, y_test)
    model_name_list.append(i)
    score_list.append(score)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_score_list.append(f1_score)
    roc_auc_list.append(roc_auc)
    
scoring_metric = pd.DataFrame({'iteration': 1,
                               'Model':model_name_list,
                               'Model_score':score_list,
                               'Model_recall':recall_list,
                               'Model_precision':precision_list,
                               'Model_f1_score':f1_score_list,
                               'Model_roc_auc':roc_auc_list})
scoring_metric['Model_name']= scoring_metric['Model'].astype('str').map(lambda x:re.sub("[\(\[].*?[\)\]]", "", x))
scoring_metric


# In[18]:


## iteration 1 : MinMax Scalar
scoring_metric


# ### Iteration 2: Standard Scalar

# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


sca=StandardScaler()

X1=pd.DataFrame(sca.fit_transform(data_x),index=data_x.index, columns=data_x.columns)
y1=data['status']


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=1,stratify = y)# Adding option stratify to get equal distributon of y in train and test
print("Target data distribution in Train")
print(y_train.value_counts(normalize=True))
print("Target data distribution in Test")
print(y_test.value_counts(normalize=True))


# In[22]:


scoring_metric_2=pd.DataFrame()
# scoring_metric=pd.DataFrame()
model_name_list= []
score_list = []
recall_list=[]
precision_list=[]
f1_score_list=[]
roc_auc_list=[]
for i in classfierdict['model']:
    score,recall,precision,f1_score,roc_auc=train_test_exp(i,x_train, x_test, y_train, y_test)
    model_name_list.append(i)
    score_list.append(score)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_score_list.append(f1_score)
    roc_auc_list.append(roc_auc)
    
scoring_metric_2 = pd.DataFrame({'iteration': 2,
                               'Model':model_name_list,
                               'Model_score':score_list,
                               'Model_recall':recall_list,
                               'Model_precision':precision_list,
                               'Model_f1_score':f1_score_list,
                               'Model_roc_auc':roc_auc_list})
scoring_metric_2['Model_name']= scoring_metric_2['Model'].astype('str').map(lambda x:re.sub("[\(\[].*?[\)\]]", "", x))
scoring_metric_2


# In[23]:


#scoring_metric=pd.DataFrame()
scoring_metric_new=scoring_metric.append(scoring_metric_2)
scoring_metric_new


# ### Iteration 3: Feature selections

# In[29]:


fs = SelectKBest(score_func=f_classif,k=5)
# apply feature selection
X_selected = fs.fit(X1, y1)
cols = fs.get_support(indices=True)
features_df_new = X1.iloc[:,cols]
var_imp=features_df_new.columns.to_list()

data_var_imp=X1[var_imp]
data_var_imp_cor=data_multicollinearity(data_var_imp)
data_var_imp_cor

X1=data_var_imp
y1=data['status']
x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=1,stratify = y)# Adding option stratify to get equal distributon of y in train and test
print("Target data distribution in Train")
print(y_train.value_counts(normalize=True))
print("Target data distribution in Test")
print(y_test.value_counts(normalize=True))


# In[30]:


scoring_metric_3=pd.DataFrame()
# scoring_metric=pd.DataFrame()
model_name_list= []
score_list = []
recall_list=[]
precision_list=[]
f1_score_list=[]
roc_auc_list=[]
name=[]
for i in classfierdict['model']:
    score,recall,precision,f1_score,roc_auc=train_test_exp(i,x_train, x_test, y_train, y_test)
    model_name_list.append(i)
    score_list.append(score)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_score_list.append(f1_score)
    roc_auc_list.append(roc_auc)

    
scoring_metric_3 = pd.DataFrame({'iteration': 3,
                               'Model':model_name_list,
                               'Model_score':score_list,
                               'Model_recall':recall_list,
                               'Model_precision':precision_list,
                               'Model_f1_score':f1_score_list,
                               'Model_roc_auc':roc_auc_list})

scoring_metric_3['Model_name']= scoring_metric_3['Model'].astype('str').map(lambda x:re.sub("[\(\[].*?[\)\]]", "", x))
scoring_metric_3


# In[31]:


scoring_metric_new=scoring_metric_new.append(scoring_metric_3)
scoring_metric_new


# In[32]:


scoring_metric_new.sort_values(by='Model_name')


# In[36]:


X=data_x
y=data['status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[37]:


from sklearn.tree import DecisionTreeClassifier

dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(X_train, y_train)


# In[38]:


print(dTree.score(X_train, y_train))
print(dTree.score(X_test, y_test))


# In[40]:


from sklearn.tree import export_graphviz

train_char_label = ['No', 'Yes']
Tree_File = open('credit_tree.dot','w')
dot_data = export_graphviz(dTree, out_file=Tree_File, feature_names = list(X_train), class_names = list(train_char_label))
Tree_File.close()


# In[41]:


from os import system
from IPython.display import Image

#Works only if "dot" command works on you machine

retCode = system("dot -Tpng credit_tree.dot -o credit_tree.png")
if(retCode>0):
    print("system command returning error: "+str(retCode))
else:
    display(Image("credit_tree.png"))


# In[71]:


import sklearn.metrics


# In[72]:


a=sklearn.metrics.SCORERS.keys()


# In[73]:


a


# In[ ]:




