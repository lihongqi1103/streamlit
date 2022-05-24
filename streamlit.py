
# from matplotlib.pyplot import text
import matplotlib.pyplot as plt
import sklearn
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time

st.title("Streamlit 超入門")

st.write("プログレスバーの表示")
"Start"

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  latest_iteration.text(f'Progress: {i}%')
  bar.progress(i+1)
  time.sleep(0.1)
st.balloons()
"Done!!!!"
left_column,right_column = st.columns(2)
button = left_column.button("右カラムに文字を表示")
if button:
  right_column.write("ここは右カラム")

expander1 = st.expander("問い合わせ1")
expander1.write("問い合わせ1回答")
expander2 = st.expander("問い合わせ2")
expander2.write("問い合わせ2回答")
expander3 = st.expander("問い合わせ3")
expander3.write("問い合わせ3回答")


text = st.text_input("あなたの趣味を教えてください。")
condition=st.slider("あなたの今の調子は？",0,100,50)

"あなたの趣味:",text
"コンディション:",condition
