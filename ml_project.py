#================================== 1st step -> DATA CLEANING ============================================
import pandas as pd

df = pd.read_csv("C:/Users/Namrata/ml/RetailDatalog.csv")
#print(df)

# ------------------ProductCategory,Outlet_Type --> removing junk values(non-alphanumeric)-----------------

import re

df['ProductCategory'] = [re.sub(r'\W', '', i) for i in df['ProductCategory']]
df['Outlet_Type'] = [re.sub(r'\W', '', i) for i in df['Outlet_Type']]
#print(df)

# -----------------------null values.... (replace/mean, sum())--------------------------------------------------

df.isnull().sum()

# -----------------------null replace by zero-------------------------------------------------------------------

df['ProductWeight'] = df['ProductWeight'].fillna(0, inplace=False)  # filling missing values
df['Outlet_Size'] = df['Outlet_Size'].fillna(0, inplace=False)  # filling missing values
df.Outlet_Size = df.Outlet_Size.astype(str)
#print(df)

# --------------------convert text to numerous---------------------------------------------------------------

from sklearn import preprocessing

l_encoder = preprocessing.LabelEncoder()  # New Object
df = df.apply(l_encoder.fit_transform)
#print(df)

# ---------------------missing values replace by mean()-------------------------------------------------------

df['Outlet_Size'] = df['Outlet_Size'].replace(0,df['Outlet_Size'].mean())  # replacing zero values with df['Outlet_Size']
df.Outlet_Size = round(df.Outlet_Size, 2)
#print(df)

# -----------------------Visualisation-- seaborn----------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.3f')
plt.show()
#DATA CLEANING is done.

#------------------------final data-> export to csv-----------------------------------------------------------------

df.to_csv("C:/Users/Namrata/ml/RetailDatalog1.csv")  #text to numorous converted data in another csv file
df.drop('ProductID', axis=1)
df.drop('Outlet-ID', axis=1)

#=========================================Hypothesis test===========================================================

from scipy.stats import ttest_ind

data = [df['ProductWeight'], df['Item_Outlet_Sales']]
stat, p = ttest_ind(df['ProductWeight'], df['Item_Outlet_Sales'])
print("p value", p)
if p < 0.05:
    print("regression model")
else:
    print("classification model")

#-------------------------LINEAR REGRESSION -> Decided by hypothesis test----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

#mapping input and target variables
X = df.drop('Item_Outlet_Sales', axis=1)
Y = df['Item_Outlet_Sales']

#--------------------------Data splitting into two parts (test_train_split)------------------------------------------
x_test, x_train, y_test, y_train = train_test_split(X, Y, test_size=0.3) #test_size must be < 0.5 always.
print(x_test)
print(x_train)
print(y_test)
print(y_train)

#==============================================*Linear Regression*===================================================

from sklearn.linear_model import LinearRegression

model = LinearRegression()
modelfit = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("prediction is:", y_pred)

ycapnew = []
for x in y_pred:
    if x < 0.5:
        x = 0
    else:
        x = 1
    ycapnew.append(x)

dfcap = pd.DataFrame(ycapnew)
print("dfcap:", dfcap)
print("y test is:",y_test)

#------------------------------Accuracy score---------------------------------------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score
print("accuracy:", accuracy_score(y_test, np.round(y_pred), normalize=False))

#----------------------------k-fold cross validation to improve accuracy------------------------------------------

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

crossycap = cross_val_predict(modelfit, df, Y, cv=4 )
score = r2_score(Y, crossycap.round())
print('k-fold score:-',score)

#----------------------------rmse----------------------------------------------------------------------------------

def rmse(y_test,ycap):
    score = np.sqrt(np.mean(((y_test-ycap)**2)))
    print("rmse score:",score)
rmse(y_test,y_pred)

#========================================Clustering (Recommendation engine)========================================

import matplotlib.pyplot as plt

plt.scatter(df['Outlet_Location_Type'], df['ProductCategory']) #scatter plot that shows k value
plt.show()

#--------------------------------------k mean cluster-------------------------------------------------------------
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3,init="k-means++")
prediction = model.fit_predict(df[["Outlet_Location_Type","ProductCategory"]])
df["cluster"]=prediction
df.to_csv("C:/Users/Namrata/ml/my_final_RetailDatalog.csv") #final data stored in csv file.
print(df)

#==================================*************************=====================================================
