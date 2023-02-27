import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn import preprocessing, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from math import log10
# from xgboost import XGBRegressor,plot_tree, plot_importance

st.set_page_config(
    page_title='LinhSenpai',
    page_icon='🤘',
)

# callback and function
def calc_slider():
    st.session_state['slider'] = st.session_state['slide_input']

def slider_input():
    st.session_state['slide_input'] = st.session_state['slider']

def convert(df):
    data = df.to_numpy()
    labelEncoder = preprocessing.LabelEncoder()
    for ind, name_type in enumerate(df.dtypes.items()):
        if (name_type[1] == 'object'):
            data[:, ind] = labelEncoder.fit_transform(data[:, ind])
    return data

def getLossValues(algorithm, df, ratio):
    data = convert(df)
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)
    if algorithm == 'Decision Tree Regression':
        reg = tree.DecisionTreeRegressor(min_samples_leaf = 4, min_samples_split = 4, random_state = 0)
        reg.fit(X_train, y_train)
    elif algorithm == 'Linear Regression':
        reg = LinearRegression()
        reg.fit(X_train, y_train)
    else:
        reg = XGBRegressor(random_state = 50, learning_rate = 0.2, n_estimators = 100)
        reg.fit(X_train, y_train)
    MAEtrain = metrics.mean_absolute_error(reg.predict(X_train), y_train)
    MAEtest = metrics.mean_absolute_error(reg.predict(X_test), y_test)
    MSEtrain = metrics.mean_squared_error(reg.predict(X_train), y_train)
    MSEtest = metrics.mean_squared_error(reg.predict(X_test), y_test)
    lossValues = [MAEtrain, MAEtest, (MSEtrain), (MSEtest)]
    lossValues = [round(value, 2) for value in lossValues ]
    log10LossValues = [log10(value) for value in lossValues ]
    return lossValues, log10LossValues

# Title
# Tên chương trình
st.sidebar.success('# Show dataframe từ file csv', icon = "📄")
st.sidebar.markdown('----')
st.title(' :orange[Chương trình] :green[kiểm tra] :violet[MAE và MSE]')
# Thêm file csv
st.title(" :red[Hãy thêm file csv vào đây]")
uploaded_file = st.file_uploader("Choose a file csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.table(df)
    # Chọn input cho bài
    st.sidebar.success('# Choose input feature', icon = "📩")
    st.sidebar.markdown('----')
    atr_choose = []
    st.title(" :blue[Choose input feature]")
    for atr in df.columns[: -1]:
        choose = st.checkbox(str(atr), True)
        if not choose:
            atr_choose.append(atr)
    if len(atr_choose) == len(df.columns[: -1]):
        st.error('You must choose least 1 feature')
    else:
        df = df.drop(columns = atr_choose)
    # Chọn thuật toán
    st.sidebar.success('# Choose Algorithm', icon = "🧮")
    st.sidebar.markdown('----')
    st.title(" :orange[Choose Algorithm]")
    algorithm = st.selectbox(
        'Hãy chọn thuật toán bạn muốn',
        ('Decision Tree Regression', 'Linear Regression', 'XGBoost'))
    st.caption('## Bạn đã chọn ' + algorithm)
    # Kéo thanh tỉ lệ
    st.sidebar.success('# Choose ratio of train/test split', icon = "🎚")
    st.sidebar.markdown('----')
    st.title(" :green[Choose ratio of train/test split]")
    st.number_input("Bạn đang chọn tỉ lệ:", 0.01, 0.99, step = 0.01, key = 'slide_input', on_change = calc_slider)
    ratio = st.slider('Chọn tỉ lệ:', 0.01, 0.99, step = 0.01, key = 'slider', on_change = slider_input)
    # get loss value
    lossValues, log10LossValues = getLossValues(algorithm, df, ratio)
    # Show biểu đồ cột
    st.sidebar.success('# Drawexplicitly chart', icon = "📊")
    st.sidebar.markdown('----')
    Loss_df = pd.DataFrame(lossValues)
    st.title(" :violet[Drawexplicitly chart]")
    df_new = {algorithm: ["MAE_train", "MAE_test", "MSE_train", "MSE_test"]}
    df_new = pd.DataFrame(df_new)
    df_new.insert(1, 'Score', log10LossValues)
    fig = px.bar(
        df_new,
        x = algorithm,
        y = "Score",
        color = algorithm,
        text = lossValues,
    )
    st.plotly_chart(fig)