## Prerequis :


Creation d un environnement virtuel et installation de toutes les bibliotheques necessaires
Les principales bibliotheques installees ainsi que leur version sont : 
-	Pandas
-	Numpy
-	Tensorflow version 1.8.1
-	Scikit-learn version 1.11.1
-	Wordcloud et PIL


## Etape 1 : Recuperation de donnees par WebScrapping 


Scrapping du site trustpilot a laide de la bibliotheque BeautifulSoup :

### - Constitution dun jeu de donnees pour creation des modeles : 

Code dans GitHub sur NoteBook: https://github.com/lille59350/satisfactionClientFinal/tree/Thomas1
« DataScientest - Projet - satisfactionClientFinal - DS - SatisfactionClients - 1 - WebScrapping.ipynb »
    o Scrapping de 120 000 commentaires cdiscount avec leur note sur le site truspilot
    o Mise en forme des donnees et enregistrement du jeu de donnees pour creation du modele en local dans un fichier cdiscount.csv

### - Constitution dun jeu de test separe qui permettra de valider notre modele sur un jeu de donnee independant.

Code dans GitHub sur NoteBook: https://github.com/lille59350/satisfactionClientFinal/tree/Thomas1 
« DataScientest - Projet - satisfactionClientFinal - DS - SatisfactionClients - 1 - WebScrapping_datatest.ipynb »

    o Scrapping de 100 commentaires amazon sur le site Trustpilot avec une repartition des notes egales :
        20 commentaires avec la note 1
        20 commentaires avec la note 2
        20 commentaires avec la note 3
        20 commentaires avec la note 4
        20 commentaires avec la note 5

    o Mise en forme des donnees et enregistrement du jeu de test en local dans un fichier amazon_test1.csv


## Etape 2 : Tokenization et nettoyage des donnees

Code dans GitHub sur le NoteBook: « DataScientest - projet - satisfactionClientFinal - DS - SatisfactionClients - 2 - nltk - regex - Wordcloud.ipynb »

### - Utilisation des bibliotheque NLTK et REGEX pour tokenizer et nettoyer le jeu de donnees

### - Chargement des stop Word pour 3 langues (Anglais, Français et Espagnol) dans une variable stop_words

### - Creation dune fonction commentaire_filtering qui :

    o Tokenize les commentaires par mots
    o Ne conserve que les mots contenant des lettres
    o Transforme tous les caracteres en minuscule
    o Supprime les stopwords dans le jeu de donnees dans 3 langues (anglais, français, espagnol) a      partir de la varibla stop_words

### - Application de la fonction commentaire_filtering sur le jeu de donnees cdiscount et enregistrement dans un fichier cdiscount2.csv

### - Creation dun 2eme fichier a partir de cdiscount2 qui categorisera les notes en sentiment negatif (valeur 0) ou positif (valeur 1) 

    o La repartition sera la suivante :
        Les notes pour lesquelles les commentaires ont des notes de 1 et 2 sont transformees en 0 qui correspondra a un sentiment negatif 
        Les notes pour lesquelles les commentaires ont des notes de 4 et 5 sont transformees en 1 qui correspondra a un sentiment positif 
        Les lignes ayant une note de 3 (sentiment ambigu) sont supprimees de ce fichier

### - Application de la fonction commentaire_filtering et enregistrement de ce nouveau jeu de donnee dans le fichier cdiscount_0_1.csv


## Etape 3 : Wordcloud


Code dans GitHub sur le NoteBook: « DataScientest - projet - SatisfactionClients - 2 - nltk - regex - Wordcloud.ipynb »
Afin davoir un aperçu des mots les plus utilises par les clients pour les commentaires negatifs et les commentaires positifs, la bibliotheque Wordcloud a ete utilisee

### - Affichage des 100 mots negatifs les plus representes dans les commentaires, sur un fond au format dune etoile

### - Affichage des 100 mots positifs les plus representes dans les commentaires, egalement sur un fond au format dune etoile


## Etape 4 : Creation des modeles de machine learning


### - Vectorisation des mots avec CountVectorizer de Sklearn avec les parametres suivants pour lensembles des modeles Sklearn testes :

    o CountVectorizer(min_df=3, max_features=5000)

#### Qualite des modeles :

- Type	Nom du modele	Parametres	Score modele sur 5 notes	Score negatifs / positifs
- Machine Learning	DecisionTreeClassifier	max_depth=10	0.69	0.90
- Machine Learning	GradientBoostingClassifier	n_estimators=25	0.64	0.90
- Machine Learning	TF_IDF RamdomForestClassifier	n_estimators=50	0.69	0.94


## Etape 5 : Creation des modeles de machine learning avec groupe de mots


On etudie maintenant les resultats que lon peut obtenir grâce differents modeles de deep learning. Le nombre max de mots dans le dictionnaire sera de 5000

### - Vectorisation des mots avec CountVectorizer de Sklearn pour des groupes de 1, 2 ou 3 mots avec les parametres suivants pour lensembles des modeles Sklearn testes :

    o CountVectorizer(max_features=5000, ngram_range=[1, 2])
    o CountVectorizer(max_features=5000, ngram_range=[2, 2])
    o CountVectorizer(max_features=5000, ngram_range=[2, 3])

#### Type Nom du modele Parametres Score modele sur 5 notes Score negatifs / positifs

- Machine Learning	RandomForestClassifier ngram[1, 2]	n_estimators=50	0.68	0.94
- Machine Learning	RandomForestClassifier ngram[2, 2]	n_estimators=50	0.64	0.91
- Machine Learning	RandomForestClassifier ngram[3, 2]	n_estimators=50	0.63	0.92


## Etape 6 : Creation des modeles de deep learning avec tensorflow


On etudie maintenant les resultats que lon peut obtenir grâce differents modeles de deep learning. Le nombre max de mots dans le dictionnaire sera de 5000
Apres etude de la distribution du nombre de mots dans chacun des commentaires, on peut choisir une valeur coherente avec : 

### - 200 mots max retenus par commentaire pour lentrainement du modele sur les notes decoupees en 5 niveaux
### - 130 mots max retenus par commentaire pour lentrainement du modele sur les notes decoupees en 2 niveaux

Pour la vectorisation des mots on utilisera la fonction texts_to_sequences puis pad_sequences de keras pour convertir la chaine de vecteur sous forme de matrice avec en nombre de colonne, le nombre maximum de mot

#### Pour le modele de deep learning, cest Embedding qui est utilise les resultats sont les suivants :
Type	        Nom du modele	parametres	                                                    Score 2 du modele sur 5 notes	Score pour sentiments negatif/positif
Deep Learning	Embedding1	    Embedding, globalAveragePooling, Dense(64)	                        0.71	                    0.95
Deep Learning	Embedding2	    Embedding, globalAveragePooling, Dense(32), Dense(32)	            0.70	                    0.95
Deep Learning	Embedding3	    Embedding, globalAveragePooling, Dense(256), Dense(128), Dense(64)	0.71	                    0.96
Deep Learning	Embedding4	    Embedding, LSTM(200)	0.70	0.95
Deep Learning	Embedding5	    Embedding, globalAveragePooling, Dense(256), Dense(128), Dense(64), Dense(32)	0.71	0.95
Deep Learning	Embedding6	    Embedding, globalAveragePooling, Dense(256)	                        0.70	                    0.96
Deep Learning	Embedding7	    Embedding, globalAveragePooling, Dense(1024)	                    0.71	                    0.96


## Etape 7 : Creation dans Streamlit dune interface web pour :


### - Page 1

    o La documentation du projet
    o Presentation des donnees utilisees pour la construction des modeles
    o Possibilite de telecharger les donnees

### - Page 2 :

    o Saisie du commentaire qui sera predit
    o Choix de la methode a utiliser
        Machine Learning
        Deep Learning

    o Choix du modele utilise pour la prediction
    o Chargement du modele selectionne (entraine auparavant et juste loader dans Streamlit)
    o Vectorisation du commentaire saisie dans streamlit
    o Application du modele a ce commentaire
    o Affichage du score du modele
    o Affichage de la prediction :
        Prediction negative ou positive
        Prediction de la note entre 1 et 5

    o Question a lutilisateur si « le resultat vous semble coherent : oui / non »
        Enregistrement de la reponse pour affichage dans un graphique au fur et a mesure


