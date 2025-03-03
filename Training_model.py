import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
mpl.style.use('ggplot')


car=pd.read_csv('quikr_car.csv') 
#loads a dataset from a CSV file named quikr_car.csv into a Pandas DataFrame called car.


car.head() 
#Is used to display the first 5 rows of the dataset stored in the variable car.
car.shape  #Check the number of rows and columns



car.info() 
#Check data types of each column


backup=car.copy() 
#creates a copy of the car DataFrame and stores it in a new variable called backup.


car=car[car['year'].str.isnumeric()]
#filters the car DataFrame by keeping only the rows 
where the 'year' column contains numeric values (not text or special characters).


car['year']=car['year'].astype(int)
#car['year']=car['year'].astype(int)


car=car[car['Price']!='Ask For Price']
#Removes rows where the 'Price' column contains the value "Ask For Price".


car['Price']=car['Price'].str.replace(',','').astype(int)
#Removes commas from the 'Price' column and converts it to an integer format.


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')
#cleans the 'kms_driven' column by:
1️⃣ Removing the "km" unit
2️⃣ Removing commas (,)
3️⃣ Extracting only the numeric value


car=car[car['kms_driven'].str.isnumeric()]
#Filters the 'car' dataset to keep only rows where 'kms_driven' contains numeric values.


car['kms_driven']=car['kms_driven'].astype(int)
#line converts the 'kms_driven' column from a string (str) format to an integer (int).


car=car[~car['fuel_type'].isna()]
#line removes rows where the 'fuel_type' column has missing (NaN) values.


car.shape
#returns the dimensions (rows and columns) of the dataset.


car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
#modifies the 'name' column by keeping only the first three words of each car's name.


car=car.reset_index(drop=True)
#This line resets the index of the DataFrame after filtering or modifying the data.
car



car.info()
#Provides a summary of the DataFrame, showing details about:
1️⃣ Number of rows and columns
2️⃣ Column names and their data types
3️⃣ Number of non-null values per column
4️⃣ Memory usage of the DataFrame


car.describe(include='all')
#command provides summary statistics for both numerical and categorical columns in your dataset.


car=car[car['Price']<6000000]
#This line filters out cars that have a price greater than or equal to ₹60,00,000 (₹6 million or ₹60 lakh).


car['company'].unique()
#This command returns a list of unique car brands (companies) in your dataset.


import seaborn as sns
#This command imports the Seaborn library, which is used for data visualization in Python.


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()
#This code creates a box plot to visualize the distribution of car prices for different car companies.


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()
#This code creates a swarm plot to visualize the distribution of car prices based on the manufacturing year.


sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)
#This code creates a scatter plot (relational plot) to show the relationship between kilometers driven (kms_driven) and car price (Price).


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)
#This code creates a box plot to compare the price distribution of cars based on their fuel type.


ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')
#This code creates a scatter plot (relational plot) to visualize car prices based on the company, fuel type, and manufacturing year.


X=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']
#This code is preparing the dataset for training a machine learning model to predict car prices.


X
#It will display the features dataset (without the Price column).


y.shape
#It will return the shape (rows, columns) of the target variable y, which contains car prices.


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#This code splits the dataset into training and testing sets so that we can train and evaluate a machine learning model properly.


from sklearn.linear_model import LinearRegression
#You're about to train a machine learning model to predict car prices using Linear Regression, a fundamental algorithm for regression problems.


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
#✅ OneHotEncoder is used to convert categorical variables (like name, company, fuel_type) into numerical values that a machine learning model can understand.
#✅ make_column_transformer allows us to apply different transformations to different columns.
#✅ handle_unknown='ignore' ensures that any unseen categories in test data won't cause an error.
#✅ A pipeline helps combine preprocessing & modeling into a single object.
✅ This makes it easier to train and test the model in a structured way.
#✅ This pipeline first applies one-hot encoding and then trains a Linear Regression model.
✅ r2_score measures how well the model predicts the target variable (Price).
✅ It ranges from 0 to 1, where 1 is a perfect prediction.
#✅ Higher r2_score means a better fit.


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
#✅ OneHotEncoder() initializes an encoder that transforms categorical variables into binary columns (0s and 1s).
✅ fit(X[['name', 'company', 'fuel_type']]) analyzes these categorical columns to learn unique categories.


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough'
#✅ OneHotEncoder(categories=ohe.categories_)
This ensures that the same category mapping (from ohe.fit()) is used, avoiding mismatches between training and testing data.
Prevents errors if the test set has fewer categories than the training set.
✅ ['name', 'company', 'fuel_type']
Specifies the categorical columns to transform.
✅ remainder='passthrough'
Keeps numerical columns (year, kms_driven) unchanged.


lr=LinearRegression()
#Now, you're initializing a Linear Regression model to predict car prices based on the preprocessed data.


pipe=make_pipeline(column_trans,lr)
#✅ make_pipeline() automates the entire ML workflow.
✅ column_trans → Preprocesses categorical variables using One-Hot Encoding.
✅ lr (LinearRegression) → Trains the model on the transformed dataset.


pipe.fit(X_train,y_train)
#Now, you're training the entire pipeline on the training data!


y_pred=pipe.predict(X_test)
#Now, you're using the trained model to predict car prices on the test dataset.


r2_score(y_test,y_pred)
#Now, you're evaluating your model’s accuracy using the R² score (coefficient of determination).


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
#Removes bias from a single train-test split.
Helps find the best random state that gives higher accuracy.
Checks if model performance remains stable across different splits.


np.argmax(scores)
#This will return the index of the highest R² score in your scores list.


scores[np.argmax(scores)]
#np.argmax(scores): Finds the index where scores is highest.
scores[...]: Retrieves the best R² score.


pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
#You're trying to predict the price of a car using your trained pipeline


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
#Splitting the data using the best random state (from np.argmax(scores)).
✅ Training a Linear Regression model inside a pipeline.
✅ Evaluating the model using R² score.


import pickle
#You’ve imported pickle, which is used for saving and loading models in Python.


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
#You've successfully saved your trained Linear Regression model using pickle! 🎉


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


pipe.steps[0][1].transformers[0][1].categories[0]
#It looks like you're trying to access the categories from the OneHotEncoder used in your ColumnTransformer.
f your model was trained with these categorical columns:
['name', 'company', 'fuel_type'],
