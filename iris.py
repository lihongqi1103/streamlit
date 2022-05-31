
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
     ('lbfgs','sgd','adam'))
text_1 = st.number_input("中間層入力")
text_2 = st.number_input("繰り返し回数入力")
if st.button('開始'):
    clf = MLPClassifier(hidden_layer_sizes=text_1,activation="activation_clf",
                    solver="solver_clf",max_iter=text_2)
    clf.fit(data_train,target_train)
    st.balloons()
    st.write("学習済み")
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(clf.loss_curve_)
    ax.legend()
    """


