import streamlit as st
import pandas as pd
import os
from txtai.embeddings import Embeddings
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import gdown


st.set_page_config(layout="wide", page_title="Satisfaction Client")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)    

st.title("Satisfaction Client")
st.sidebar.image("images/OIP.jfif")
st.sidebar.markdown("Developed by Yvon-Arnaud GBE and Thomas FOURTOUILL using [Streamlit](https://www.streamlit.io)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 0.0.2")
query = st.sidebar.text_input("Query")
num_results = st.sidebar.number_input("Number of Results", 1, 2000, 20)
ignore_search_words = st.sidebar.checkbox("Ignore Search Words")

with st.expander("How to Use This"): 
    st.write(Path("README.md").read_text())


st.markdown("les 100 mots négatifs les plus représentés sont présents dans l'images ci-dessous")
st.image("./images/cloud_negatif.png")
st.write("les 100 mots positifs les plus représentés sont présents dans l'images ci-dessous")
st.image("./images/cloud_negatif.png")

cdiscount = pd.read_csv('./cdiscount.csv')
X = cdiscount['commentaire']

#fig = plt.hist(cdiscount['note'])
#st.pyplot(plt.hist(cdiscount['note']))
#arr = np.random.normal(1, 1, size=100)

fig, ax = plt.subplots()
ax.hist(cdiscount['note'], bins=20)

st.pyplot(fig)




def create_html(result):
    output = f""
    spans = []
    for token, score in result["tokens"]:
        color = None
        if score >= 0.1:
            color = "#fdd835"
        elif score >= 0.075:
            color = "#ffeb3b"
        elif score >= 0.05:
            color = "#ffee58"
        elif score >= 0.02:
            color = "#fff59d"

        spans.append((token, score, color))

    if result["score"] >= 0.05 and not [color for _, _, color in spans if color]:
        mscore = max([score for _, score, _ in spans])
        spans = [(token, score, "#fff59d" if score == mscore else color) for token, score, color in spans]

    for token, _, color in spans:
        if color:
            output += f"<span style='background-color: {color}'>{token}</span> "
        else:
            output += f"{token} "
    return output




if st.sidebar.button("Search"):
    "hello word"

