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

st.title("Streamlit 学習")
st.subheader("アヤメのデータ学習")
image_00 = Image.open('iris01.png')
st.image(image_00, caption='アヤメのデータ分類')

iris = datasets.load_iris()

df_iris = pd.DataFrame(data = iris.data,columns=iris.feature_names)

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

activation_clf = st.radio(
     "activation 活性化関数",
     ('identity','logistic','tanh','relu'))
if activation_clf=='identity':
    st.latex('''f(x)=x''')
elif activation_clf=='logistic':
    st.latex('''f(x)=\cfrac{1}{1+e^{-x}}''')
elif activation_clf=='tanh':
    st.latex('''f(x)=\cfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}''')
elif activation_clf=='relu':

    st.latex('''f(x) = \left\{
\begin{array}{ll}
1 & (x \geq 0)\\
0 & (x &lt; 0)
\end{array}
\right.
''')

solver_clf = st.radio(
     "solver 最適化手法",
     ('sgd','adam'))

text_1=st.slider("中間層入力",0,100,10,1)
text_2=st.slider("繰り返し回数入力",0,1000,100,1)

start = st.radio(
     "学習開始",
     ('YES','NO'))
if start=="YES":
    @st.cache
    def return_model(text_1, activation_clf, solver_clf, text_2):
        clf = MLPClassifier(hidden_layer_sizes=text_1,activation= activation_clf,
                        solver= solver_clf , max_iter=text_2)
        clf.fit(data_train,target_train)
        return clf

    clf = return_model(text_1, activation_clf, solver_clf, text_2)
    
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
    data, label = iris.data, iris.target

    fig = plt.figure()
    ax = fig.add_subplot()
    a = {"x" : [], "y" : []}
    b = {"x" : [], "y" : []}
    c = {"x" : [], "y" : []}
    for d, l in zip(data, label):
        if l == 0:
            a["x"].append(d[0] * d[1])
            a["y"].append(d[2] * d[3])

        elif l == 1:
            b["x"].append(d[0] * d[1])
            b["y"].append(d[2] * d[3])

        else:
            c["x"].append(d[0] * d[1])
            c["y"].append(d[2] * d[3])
            
    ax.scatter(a["x"], a["y"], color = "red", label = "0　Setosa")
    ax.scatter(b["x"], b["y"], color = "blue", label = "1　Versicolor")
    ax.scatter(c["x"], c["y"], color = "green", label = "2　Versinica")
    ax.scatter(data_0 *data_1, data_2*data_3 , color = "black", label = "mine")
    ax.legend()
    st.pyplot(fig)
