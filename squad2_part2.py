import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
st.set_page_config(
    page_title='LinhSenpai',
    page_icon='🤘',
)
st.set_option('deprecation.showPyplotGlobalUse', False)
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
# Title
# Tên chương trình
st.title(' :violet[Classification Algorithms]')
# Thêm file csv
st.title(" :red[Hãy thêm file csv vào đây]")
uploaded_file = st.file_uploader("Choose a file csv")
if uploaded_file:
    error = 0
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    # Chọn input cho bài
    atr_not_choose = []
    st.title(" :blue[Choose input feature]")
    for atr in df.columns[: -1]:
        choose = st.checkbox(str(atr))
        if not choose:
            atr_not_choose.append(atr)
    if len(atr_not_choose) != len(df.columns[: -1]):
        df = df.drop(columns = atr_not_choose)
    else:
        st.error('You must choose least 1 feature')
        error = 1
        # Chọn thuật toán
    if not error:
        col1, col2 = st.columns(2)
        with col1:
            st.title(" :orange[Choose Algorithm]")
            algorithm = st.selectbox(
                'Hãy chọn thuật toán bạn muốn',
                ('Decision Tree Classification', 'Linear Classification', 'Logistic Regression', 'Navie Bayes', 'Random Forest'))
            st.caption('## Bạn đã chọn ' + algorithm)
        with col2:
        # Kéo thanh tỉ lệ
            st.title(" :green[Choose ratio of test]")
            st.number_input("Bạn đang chọn tỉ lệ:", 0.01, 0.99, step = 0.01, key = 'slide_input', on_change = calc_slider)
            ratio = st.slider('Chọn tỉ lệ:', 0.01, 0.99, step = 0.01, key = 'slider', on_change = slider_input)
        # get loss value
        data = convert(df)
        X = data[:, :-1]
        y = data[:, -1].astype('int')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)
        if algorithm == 'Decision Tree Classification':
            clf = tree.DecisionTreeClassifier(random_state=0, max_depth=2)
            clf.fit(X_train, y_train)
        elif algorithm == 'Linear Classification':
            clf = SGDClassifier(max_iter=1000, tol=1e-3)
            clf.fit(X_train, y_train)
        elif algorithm == 'Logistic Regression':
            clf = LogisticRegression(C=1e5, solver = 'lbfgs', multi_class = 'multinomial')
            clf.fit(X_train, y_train)
        elif algorithm == 'Navie Bayes':
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
        else:
            clf = RandomForestClassifier(n_estimators = 100, 
                                criterion ='gini', 
                                max_depth = 3, 
                                min_samples_split = 2, 
                                min_samples_leaf = 15,)
            clf.fit(X_train, y_train)
        F1Train = metrics.f1_score(clf.predict(X_train), y_train, average = 'macro')
        F1Test = metrics.f1_score(clf.predict(X_test), y_test, average = 'macro')
        accuracy_values = [F1Train, F1Test]
        accuracy_values = np.array([round(accuracy_value*100, 2) for accuracy_value in accuracy_values ])
        cm = metrics.confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
        # Show biểu đồ cột
        st.title(" :violet[Drawexplicitly chart]")
        labels = np.array(['F1 Score Train', 'F1 Score Test'])
        fig, ax = plt.subplots()
        ax.bar(labels, accuracy_values, 0.6, 0.01)
        ax.set_xticks(labels)
        ax.set_yticks(range(0, 101, 10))
        plt.xlabel(algorithm)
        plt.ylabel('F1 Score (%)')
        for ind,val in enumerate(accuracy_values):
            plt.text(ind, val + 0.6, str(val), transform = plt.gca().transData,horizontalalignment = 'center', color = 'red',fontsize = 'small')
        st.pyplot(fig)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        st.pyplot()