# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:30:02 2023

@author: t.fourtouill
"""

import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model
from joblib import dump, load

pd.options.display.max_colwidth=800
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

cdiscount = pd.read_csv('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/cdiscount2.csv')
cdiscount = cdiscount[cdiscount['commentaire'].isna()==False]
X = cdiscount['commentaire']
    
# instanciation des pages ici 3
page = st.sidebar.radio(label='page n°', options=[1, 2, 3])

# contenu de la 1ère page
if page==1:
    
    st.title("présentation du projet de prédiction de la satisfaction clients")
    st.markdown(""" 
                l'objectif est de prédire la note à attribuer en fonction du commentaires client
                """)
    st.write("voici un aperçu du jeu de données qui a servi à la construction des modèles : ")
    
    with st.spinner(text="Les données sont en cours de chargement ...."):
        st.dataframe(cdiscount)
   
    # mise en cache du DataFrame cdiscount qui a servi à construire les modèles pour téléchargement dans stramlit
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(cdiscount)
    st.download_button(label="Download", data=csv, file_name="commentaire_cdiscount.csv", help="si vous souhaitez télécharger le fichier qui contient les 120000 commentaires cdiscount avec les notes qui on servies à construire les modèles de prédiction, cliquer sur ce bouton")
    
    st.write("")
    st.write("la répartition des notes dans notre jeu de données :")
    barre = cdiscount['note'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax1 = plt.subplot(1,1,1)
    ax1.bar(barre.index, barre)
    plt.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write("")
    st.markdown("""
                Affichons les mots le plus fréquement contenus dans les commentaires négatifs et dans les commentaires positifs
                """)
    
    from PIL import Image
    image = Image.open('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/images/cloud_negatif.png')
    st.image(image, caption=' les 100 mots négatifs les plus représentés')
    image = Image.open('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/images/cloud_positif.png')
    st.image(image, caption=' les 100 mots positifs les plus représentés')
   

# contenu de la 2ème page
elif page==2:
   
    st.markdown("""
                Merci de saisir le commentaire à prédire
                """)
    
    # saisie du commentaire
    comment = st.text_input("saisir un commentaire :")
    
    # Vectorization du commentaire pour les modèles de Machine Learning
    comment2 = [comment]
       
    vectorizer = CountVectorizer(max_features=10000, min_df=3)
    X_train = vectorizer.fit_transform(X).todense()
    comment_to_predict = vectorizer.transform(comment2).todense()
    
    # Vectorization du commentaire pour les modèles de Deep Learning
    # tokenisation des commentaires
    num_words = 1000
    tk = Tokenizer(num_words=num_words, lower=True)
    
    # entrainement de la tokenisation du commentaire user à prédire
    tk.fit_on_texts(comment)
    
    # vectorisation des token
    check_seq = tk.texts_to_sequences(comment)
    
    # mise sous matrice numpy
    max_words = 130
    check_pad = pad_sequences(check_seq, maxlen=max_words, padding='post')
    
    st.markdown("""
                    affichage de la vectorisation du commentaire test
                    """)
    
    st.write(comment_to_predict)
    st.write(check_pad)
   
    st.markdown("""
                Merci de sélectionner le modèle que vous souhaitez
                appliquer sur le dataset
                """)
    
    notation = st.radio(label = "Choisissez le type de prédictions :",
             options = ["notation de 1 à 5",
                        "Commentaire négatif ou positif"])
    
    if notation == "notation de 1 à 5":    
    
        selection_method = st.radio(label = "choisissez la méthode :",
                 options = ["Machine Learning",
                            "Deep Learning"])
     
        if selection_method == "Machine Learning":
            
            selection_model_ML = st.radio(label = "choisissez un modèle de machine learning à évaluer :",
                     options = ["RandomForest",
                                "DecisionTreeClassifier",
                                "GradientBoostingClassifier"
                                ])      
        
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=5000)
            comment_to_predict = vectorizer.fit_transform(comment_to_predict).todense()
        
            if selection_model_ML == "RandomForest":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_rf.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
        
            if selection_model_ML == "DecisionTreeClassifier":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_dtc.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
        
            if selection_model_ML == "GradientBoostingClassifier":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_gbc.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
    
        if selection_method == "Deep Learning":
        
            selection_model_DL = st.radio(label = "choisissez un modèle de deep learning à évaluer :",
                     options = ["Embedding1",
                                "Embedding2",
                                "Embedding3",
                                "Embedding4",
                                "Embedding5",
                                "Embedding6"
                                ])
            
            if selection_model_DL == "Embedding1":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_embedding1')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
                
            if selection_model_DL == "Embedding2":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_embedding2')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
                  
            if selection_model_DL == "Embedding3":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_embedding3')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
        
            if selection_model_DL == "Embedding4":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_embedding4')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
        
            if selection_model_DL == "Embedding5":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_embedding5')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
        
            if selection_model_DL == "Embedding6":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_embedding6')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
            
    if notation == "Commentaire négatif ou positif":    
    
        selection_method = st.radio(label = "choisissez la méthode :",
                 options = ["Machine Learning",
                            "Deep Learning"])
     
        if selection_method == "Machine Learning":
            
            selection_model_ML = st.radio(label = "choisissez un modèle de machine learning à évaluer :",
                     options = ["RandomForest",
                                "RandomForestTFIDF",
                                "DecisionTreeClassifier",
                                "GradientBoostingClassifier",
                                "RandomForest_ngram_1_2",
                                "RandomForest_ngram_2_2"
                                ])      
        
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=5000)
            comment_to_predict = vectorizer.fit_transform(comment_to_predict).todense()
        
            if selection_model_ML == "RandomForest":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_rf_0_1.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
        
            if selection_model_ML == "RandomForestTFIDF":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_rf_0_1_tfidf.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
        
            if selection_model_ML == "DecisionTreeClassifier":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_dtc_0_1.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
        
            if selection_model_ML == "GradientBoostingClassifier":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_gbc_0_1.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)
    
            if selection_model_ML == "RandomForest_ngram_1":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_rf_0_1_ngrams_1_2.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)    
                
            if selection_model_ML == "RandomForest_ngram_2":
                model = load('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_rf_0_1_ngrams_2_2.joblib')
                check_predict = model.predict(comment_to_predict)
                st.write(check_predict)  

        if selection_method == "Deep Learning":
        
            selection_model_DL = st.radio(label = "choisissez un modèle de deep learning à évaluer :",
                     options = ["Embedding1",
                                "Embedding2",
                                "Embedding3",
                                "Embedding4",
                                "Embedding5",
                                "Embedding6"
                                ])
            
            if selection_model_DL == "Embedding1":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_0_1_embedding1')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
                
            if selection_model_DL == "Embedding2":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_0_1_embedding2')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
                  
            if selection_model_DL == "Embedding3":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_0_1_embedding3')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
        
            if selection_model_DL == "Embedding4":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_0_1_embedding4')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
        
            if selection_model_DL == "Embedding5":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_0_1_embedding5')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)
            
            if selection_model_DL == "Embedding6":
                model = tf.keras.models.load_model('C:/Users/t.fourtouill/Bureau/streamlit_satisfaction_clients/models/model_0_1_embedding6')
                check_predict = model.predict(check_pad, verbose=1)
                check_predict_class = check_predict.argmax(axis=1)
                st.write(check_predict_class)





