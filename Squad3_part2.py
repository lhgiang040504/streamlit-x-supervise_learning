import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def convert(df):
    data = df.to_numpy()
    labelEncoder = preprocessing.LabelEncoder()
    for ind, name_type in enumerate(df.dtypes.items()):
        if (name_type[1] == 'object'):
            data[:, ind] = labelEncoder.fit_transform(data[:, ind])
    return data

st.title(':red[SQUAD_3] :coffee:')
st.markdown(":one: **_:blue[Upload and show Dataframe]_** :waxing_crescent_moon:")
upload_file = st.file_uploader("Choose a CSV file")

if (upload_file is not None):
    df = pd.read_csv(upload_file)  
    st.dataframe(df)
    
    #clean data
    df.dropna(inplace = True)
    le = LabelEncoder()
    is_Category = df.dtypes == object 
    category_column_list = df.columns[is_Category].tolist()
    df[category_column_list] = df[category_column_list].apply(lambda col: le.fit_transform(col)) # biến object thành số

    #chosse features
    st.markdown(":two: **_:blue[Choose Input Feature]_**")
    st.write("What columns do you want to use for training", str(df.columns[-1]), ':first_quarter_moon:')
    choice = []
    run = False
    for i in range(0, len(df.columns) - 1):        
        if (st.checkbox(df.columns[i]) == True):
            choice.append(i)
            run = True   
    df = df.iloc[:, choice]
    st.dataframe(df)
    
    # chosse algorithm
    if (run): 
        st.markdown(":three: **_:blue[Choose Algorithm]_**")
        algorithm = st.selectbox(
            "Choose one of three algorithms for training :waxing_gibbous_moon::",
            ('Logictic Regression', 'Random Forest', 'Navie Bayes')
            ) 

        st.markdown(":four: **_:blue[Drawing explicity chart]_**")
        
        #chosse train/test ratio
        ratio = st.slider('Choose train size :full_moon::', 0.0, 1.0, 0.25)
        if (ratio == 1 or ratio == 0):
            st.markdown("**Choose another value please**")
        else:
            data = convert(df)
            x = data[:, :-1]
            y = data[:, -1].astype('int')
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio)
            
            #calculate model
            y_pred = []
            if (algorithm == 'Logictic Regression') :
                model = LogisticRegression()
            if (algorithm == 'Random Forest') :
                model = RandomForestClassifier()        
            if (algorithm == 'Navie Bayes') :
                model = MultinomialNB()

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)    
            
            
            f1_train = metrics.f1_score(model.predict(x_train), y_train, average = "macro")
            f1_test = metrics.f1_score(model.predict(x_test), y_test, average = "macro")

            accuracy_values = [f1_train, f1_test]
            accuracy_values = np.array([round(accuracy_value*100, 2) for accuracy_value in accuracy_values ])
            
            #show chart
            fig, ax = plt.subplots()
            ax.bar(["f1_train", "f1_test"], accuracy_values, 0.6, 0.01)
            ax.set_xticks(["f1_train", "f1_test"])
            ax.set_yticks(range(0, 101, 10))
            plt.xlabel(algorithm)
            plt.ylabel('F1 Score (%)')
            
            #show metrics chart
            cm = metrics.confusion_matrix(y_test, y_pred, labels = model.classes_)
            st.pyplot(fig)

            disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=model.classes_)
            disp.plot()
            st.pyplot()