# from matplotlib.pyplot import text
import matplotlib.pyplot as plt
import sklearn
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


st.title("アヤメのデータ学習")

st.write("学習")


iris = datasets.load_iris()

df_iris = pd.DataFrame(data = iris.data,columns=iris.feature_names)

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

activation_clf = st.radio(
     "activation",
     ('identity','logistic','tanh','relu'))
solver_clf = st.radio(
     "solver",
     ('sgd','adam'))
#text_1 = st.number_input("中間層入力")
text_1=st.slider("中間層入力",0,100,10,1)
text_2=st.slider("繰り返し回数入力",0,1000,100,1)
#text_2 = st.number_input("繰り返し回数入力")
"""
if st.button("学習開始"):
    clf = MLPClassifier(hidden_layer_sizes=text_1,activation= activation_clf,
                    solver= solver_clf ,max_iter=text_2)
    clf.fit(data_train,target_train)
    st.balloons()
    st.write("学習済み")
    st.write("損失関数")
    st.line_chart(clf.loss_curve_)

    data_0 = st.slider("ガクの長さ入力",0.0,10.0,0.0,0.01)
    data_1 = st.slider("ガクの幅入力",0.0,10.0,0.0,0.01)
    data_2 = st.slider("花弁の長さ入力",0.0,10.0,0.0,0.01)
    data_3 = st.slider("花弁の幅入力",0.0,10.0,0.0,0.01)
    if st.button("識別開始"):
        data_test=[[data_0,data_1,data_2,data_3],]
        st.write(clf.predict(data_test))
"""
@st.cache
def test_check(a = False):
    if not a: return False
    else: return True
a = test_check()
st.write(a)
if a or test_check():
    
    a = test_check()
    
    @st.cache
    def return_model(text_1, activation_clf, solver_clf, text_2):
        clf = MLPClassifier(hidden_layer_sizes=text_1,activation= activation_clf,
                        solver= solver_clf , max_iter=text_2)
        clf.fit(data_train,target_train)
        return clf

    clf = return_model(text_1, activation_clf, solver_clf, text_2)
    st.balloons()
    st.write("学習済み")
    st.write("損失関数")
    st.line_chart(clf.loss_curve_)

    data_0 = st.slider("ガクの長さ入力",0.0,10.0,0.0,0.01)
    data_1 = st.slider("ガクの幅入力",0.0,10.0,0.0,0.01)
    data_2 = st.slider("花弁の長さ入力",0.0,10.0,0.0,0.01)
    data_3 = st.slider("花弁の幅入力",0.0,10.0,0.0,0.01)
    if st.button("識別開始"):
        data_test=[[data_0,data_1,data_2,data_3],]
        st.write(clf.predict(data_test))
