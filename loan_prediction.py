#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#%matplotlib inline

#Extracting the csv files

df_train = pd.read_csv('train.csv')
#print(df_train.head())
#print(df_train.shape)

df_test = pd.read_csv('test.csv')
#print(df_test.head())
#print(df_test.shape)


#print(df_train.isnull().sum())



#finding null values in the train dataset

print(df_train.isnull().sum())

#info of train dataset

print(df_train.info())

#print(df_train['Loan_Status'].head(25))
#replacing all missing values

df_train['Gender'] = df_train['Gender'].fillna(df_train['Gender'].dropna().mode().values[0] )

df_train['Married'] = df_train['Married'].fillna(df_train['Married'].dropna().mode().values[0] )

df_train['Dependents'] = df_train['Dependents'].fillna(df_train['Dependents'].dropna().mode().values[0] )

df_train['Self_Employed'] = df_train['Self_Employed'].fillna(df_train['Self_Employed'].dropna().mode().values[0] )

df_train['LoanAmount'] = df_train['LoanAmount'].fillna(df_train['LoanAmount'].dropna().median() )

df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].dropna().mode().values[0] )

df_train['Credit_History'] = df_train['Credit_History'].fillna(df_train['Credit_History'].dropna().mode().values[0] )

#Exploratory Analysis
sns.countplot(y='Gender', hue='Loan_Status', data=df_train)
plt.title('Gender and Loan_Status countplot')
plt.savefig('gender_and_load_status_countplot.png')

sns.countplot(y='Married', hue='Loan_Status', data=df_train)
plt.title('Married and Loan_Status countplot')
plt.savefig('married_and_load_status_countplot.png')

sns.countplot(y='Self_Employed', hue='Loan_Status', data=df_train)
plt.title('Self_Employed and Loan_Status countplot')
plt.savefig('self_employed_and_load_status_countplot.png')

sns.countplot(y='Credit_History', hue='Loan_Status', data=df_train)
plt.title('Credit_History and Loan_Status countplot')
plt.savefig('credit_history_and_load_status_countplot.png')

sns.countplot(y='Property_Area', hue='Loan_Status', data=df_train)
plt.title('Property_Area and Loan_Status countplot')
plt.savefig('property_area_and_load_status_countplot.png')

sns.countplot(y='Loan_Amount_Term', hue='Loan_Status', data=df_train)
plt.title('Loan_Amount_Term and Loan_Status countplot')
plt.savefig('loan_amount_term_and_load_status_countplot.png')

grid = sns.FacetGrid(df_train, row='Gender', col='Married', size=2.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of gender,married and ApplicantIncome')
plt.savefig('histogram_of_gender_married_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Gender', col='Education', size=2.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of gender,education and ApplicantIncome')
plt.savefig('histogram_of_gender_education_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Married', col='Education', size=2.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of married,education and ApplicantIncome')
plt.savefig('histogram_of_married_education_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Self_Employed', col='Education', size=2.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of self_employed,education and ApplicantIncome')
plt.savefig('histogram_of_self_employed_education_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Married', col='Dependents', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of married,dependents and ApplicantIncome')
plt.savefig('histogram_of_married_dependents_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Self_Employed', col='Dependents', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of self_employed,dependents and ApplicantIncome')
plt.savefig('histogram_of_self_employed_dependents_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Gender', col='Dependents', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of gender,dependents and ApplicantIncome')
plt.savefig('histogram_of_gender_dependents_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Education', col='Dependents', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of education,dependents and ApplicantIncome')
plt.savefig('histogram_of_education_dependents_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Property_Area', col='Dependents', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of property_area,dependents and ApplicantIncome')
plt.savefig('histogram_of_property_area_dependents_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Married', col='Credit_History', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of married,credit_history and ApplicantIncome')
plt.savefig('histogram_of_married_credit_history_and_ApplicantIncome.png')

grid = sns.FacetGrid(df_train, row='Education', col='Credit_History', size=3.2, aspect=1.6)
grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
grid.add_legend()
plt.title('histogram of education,credit_history and ApplicantIncome')
plt.savefig('histogram_of_education_credit_history_and_ApplicantIncome.png')


#changing object datatypes to float for float
gender = {'Male': 1,'Female': 2}
df_train['Gender'] = [gender[item] for item in df_train['Gender'] ]

married = {'Yes': 1,'No': 2}
df_train['Married'] = [married[item] for item in df_train['Married'] ]

dependents = {'0': 0,'1': 1, '2': 2, '3+': 3}
df_train['Dependents'] = [dependents[item] for item in df_train['Dependents'] ]

education = {'Graduate': 1,'Not Graduate': 2}
df_train['Education'] = [education[item] for item in df_train['Education'] ]

self_employed = {'Yes': 1,'No': 2}
df_train['Self_Employed'] = [self_employed[item] for item in df_train['Self_Employed'] ]

loan_status = {'Y': 1,'N': 2}
df_train['Loan_Status'] = [loan_status[item] for item in df_train['Loan_Status'] ]

property_area = {'Urban': 0,'Semiurban': 1, 'Rural': 2}
df_train['Property_Area'] = [property_area[item] for item in df_train['Property_Area'] ]

#to show outliers

df_train.boxplot()
#plt.show()

#clear outliers
print(df_train['ApplicantIncome'].quantile(0.10))
print(df_train['ApplicantIncome'].quantile(0.90))

print(df_train['CoapplicantIncome'].quantile(0.10))
print(df_train['CoapplicantIncome'].quantile(0.90))

df_train['ApplicantIncome'] = np.where(df_train['ApplicantIncome'] <2216.0, 2216.0,df_train['ApplicantIncome'])
df_train['ApplicantIncome'] = np.where(df_train['ApplicantIncome'] >9459.0, 9459.0,df_train['ApplicantIncome'])

#print(df_train['ApplicantIncome'].head(20))

#df_train['CoapplicantIncome'] = np.where(df_train['ApplicantIncome'] <2216.0, 2216.0,df_train['ApplicantIncome'])
df_train['CoapplicantIncome'] = np.where(df_train['CoapplicantIncome'] >3782.0, 3782.0,df_train['CoapplicantIncome'])

#print(df_train['CoapplicantIncome'].head(20))

print(df_train.corr())

df_train.boxplot()
#plt.show()


#finding null values in the test dataset
print(df_test.isnull().sum())

#replacing all missing values in the test dataset

df_test['Gender'] = df_test['Gender'].fillna(df_test['Gender'].dropna().mode().values[0] )

df_test['Dependents'] = df_test['Dependents'].fillna(df_test['Dependents'].dropna().mode().values[0] )

df_test['Self_Employed'] = df_test['Self_Employed'].fillna(df_test['Self_Employed'].dropna().mode().values[0] )

df_test['LoanAmount'] = df_test['LoanAmount'].fillna(df_test['LoanAmount'].dropna().median() )

df_test['Loan_Amount_Term'] = df_test['Loan_Amount_Term'].fillna(df_test['Loan_Amount_Term'].dropna().mode().values[0] )

df_test['Credit_History'] = df_test['Credit_History'].fillna(df_test['Credit_History'].dropna().mode().values[0] )


gender = {'Male': 1,'Female': 2}
df_test['Gender'] = [gender[item] for item in df_test['Gender'] ]

married = {'Yes': 1,'No': 2}
df_test['Married'] = [married[item] for item in df_test['Married'] ]

dependents = {'0': 0,'1': 1, '2': 2, '3+': 3}
df_test['Dependents'] = [dependents[item] for item in df_test['Dependents'] ]

education = {'Graduate': 1,'Not Graduate': 2}
df_test['Education'] = [education[item] for item in df_test['Education'] ]

self_employed = {'Yes': 1,'No': 2}
df_test['Self_Employed'] = [self_employed[item] for item in df_test['Self_Employed'] ]

property_area = {'Urban': 0,'Semiurban': 1, 'Rural': 2}
df_test['Property_Area'] = [property_area[item] for item in df_test['Property_Area'] ]


#defining the independent features for the test dataset
test = df_test.iloc[:,1:12]#independent features (excluded the first column
#ie Loan_ID because Loan_Status does not depend on it)
test_Loan_ID = df_test['Loan_ID']
#Splitting the train dataset into dependent and independent features

X = df_train.iloc[:,1:12] #independent features (excluded the first column
#ie Loan_ID because Loan_Status does not depend on it)
y = df_train["Loan_Status"] #dependent variable
print(X.head())



#training the algorithm using logistic regression

regressor = LogisticRegression()  
regressor.fit(X, y)

#make prediction using the test data

y_pred = regressor.predict(test)

#view predicted loan status 
pred_Loan_Status = pd.DataFrame({'Loan_ID':test_Loan_ID, 'Predicted': y_pred})
#print(pred_Loan_Status.head(20))

predloan_status = {1: 'Y',2: 'N'}
pred_Loan_Status['Predicted'] = [predloan_status[item] for item in pred_Loan_Status['Predicted'] ]

print(pred_Loan_Status.head(20))
print(pred_Loan_Status.shape)


