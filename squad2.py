import streamlit as st
 
import pandas as pd
import csv
import math
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#Set up Page
st.set_page_config(
  page_title = "Applications of Regression", 
  page_icon = "",
  layout = "wide",
  initial_sidebar_state="expanded" 
  )
#Design Page
st.title("Regression with Dataframe")
col_1, col_2 = st.columns((2.5,2))
with col_1:
  #Upload csv file to your browse
  st.header("Choose a file")
  uploaded_file = st.file_uploader('')
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
with col_2:
  #Choose Algorithm
  checkbox_names = ["Decision Tree", "Linear Regression", "XGBoost"]
  st.header("Choose Algorithm")
  checkbox = st.radio('', checkbox_names)
  #Choose test size
  st.header("Text test size")
  test_size = st.number_input("Text test size")
  if uploaded_file is not None:
    #Select feature
    st.header("Choose features: ")
    keys = df.keys()[ : -1]
    options = st.multiselect('', keys)
    #importing datasets  
    data_set = df
    #Extracting Independent and dependent Variable
    if options != []:
      x= data_set.loc[ : , options].values  
      y= data_set.iloc[ : , -1].values
      #Catgorical data  
      labelencoder_x= LabelEncoder()
      index_key = -1
      for key in options:
          index_key = index_key + 1
          if df.dtypes[key] == 'object':
            x[ : , index_key]= labelencoder_x.fit_transform(x[ : , index_key])
      #Splitting the dataset into training and test set.
      if test_size > 0 and test_size <= 1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 0)
        #Run choosen Regression Model:
        if checkbox == "Linear Regression":
          #Fitting the MLR model to the training set:  
          regressor= LinearRegression()  
        elif checkbox == "XGBoost":
          # Train model bằng XGB
          regressor = xgb.XGBRegressor(random_state=42, n_estimators = 100)
        elif checkbox == "Decision Tree":
          #Fitting the Decision Tree model to the training set:
          regressor = DecisionTreeRegressor(min_samples_leaf = 4, min_samples_split = 4, random_state=0)
        #Fitting model and data
        regressor.fit(x_train, y_train)
        #Predicting the Test set result;  
        y_pred= regressor.predict(x_test) 
        y_pred_train = regressor.predict(x_train)
        #Show detail and bar chart
        keys = ["MAE(train)", "MAE(test)", "MSE(train)", "MSE(test)"]
        values = [math.log(mean_absolute_error(y_test, y_pred), 10), math.log(mean_absolute_error(y_train, y_pred_train), 10), 
        math.log(mean_squared_error(y_test, y_pred), 10), math.log(mean_squared_error(y_train, y_pred_train), 10)]
        #Show bar chart
        st.subheader("Bar chart (log)")
        chart_data = pd.DataFrame(values, keys)
        st.bar_chart(chart_data)
#Show details
if ((uploaded_file is not None) and (options != []) and (test_size <= 1 and test_size > 0)) :
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("MAE(test)", round(math.log(mean_absolute_error(y_train, y_pred_train), 10), 5))
  col2.metric("MAE(train)", round(math.log(mean_absolute_error(y_test, y_pred), 10),5))
  col3.metric("MSE(test)", round(math.log(mean_squared_error(y_train, y_pred_train), 10), 5))
  col4.metric("MSE(train)", round(math.log(mean_squared_error(y_test, y_pred), 10), 5))