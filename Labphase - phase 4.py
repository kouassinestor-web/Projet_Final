# -*- coding: utf-8 -*-
"""Le NLP, ou Traitement du Langage Naturel, est une branche de l'intelligence artificielle qui permet aux machines de comprendre, d'interpréter et de générer du langage humain.Le défi : Contrairement aux données structurées (chiffres), le langage humain est ambigu, rempli de fautes, d'ironie et de variations (SMS, soutenu).Le but : Transformer un texte brut en une donnée mathématique que l'ordinateur peut traiter.

Un chatbot moderne ne se contente pas de répondre à des mots-clés isolés ; il suit un processus intelligent:
Analyse de l'intention (NLU) : Le chatbot identifie ce que l'utilisateur veut (ex: "Où est mon paquet ? Intention : Suivi de colis).

Extraction d'entités : Il repère les informations clés (ex: "n° 12345").Gestion du dialogue : Il choisit la réponse la plus adaptée ou pose une question complémentaire.

La bibliothèque NLTK (Natural Language Toolkit) est la "boîte à outils" que nous avons utilisée pour transformer nos 1500 messages en modèles de décision.
Elle permet :La Tokenisation, Le Nettoyage et Le Stemming
La Classification : Utiliser des algorithmes (Naive Bayes, Arbre de Décision) pour prédire la catégorie d'un message inconnu sur la base de cet apprentissage.

Pour illustrer notre  présentation, nous allons maintenant simuler le fonctionnement réel de notre modèle optimisé."
"""

import pandas as pd
import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import classify

# Téléchargement des ressources nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# 1. Initialisation des outils de prétraitement
stemmer = SnowballStemmer('french')
stop_words = set(stopwords.words('french'))

def extraire_features(phrase):
    """Nettoie le texte et extrait les racines des mots (stems)"""
    if not isinstance(phrase, str): return {}
    tokens = word_tokenize(phrase.lower(), language='french')
    # On garde les mots alphabétiques, non stop-words, et on applique le stemming
    stems = [stemmer.stem(m) for m in tokens if m.isalpha() and m not in stop_words]
    return {mot: True for mot in stems}

# 2. Chargement du dataset
# Assurez-vous que le chemin du fichier est correct
path = "/content/messages_support_clients_france.csv"
df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')

# 3. Préparation des données pour NLTK
dataset_complet = []
for index, row in df.iterrows():
    dataset_complet.append((extraire_features(row['Message']), row['Catégorie']))

# Mélange et split (80% train / 20% test)
random.seed(42)
random.shuffle(dataset_complet)
taille_split = int(len(dataset_complet) * 0.8)
train_set = dataset_complet[:taille_split]
test_set = dataset_complet[taille_split:]

# 4. Entraînement de l'Arbre de Décision (Ajusté)
print("Entraînement de l'Arbre de Décision en cours...")
dt_tuned = nltk.DecisionTreeClassifier.train(
    train_set,
    entropy_cutoff=0.05,
    support_cutoff=15
)

# 5. Évaluation
accuracy = classify.accuracy(dt_tuned, test_set)
print("\n" + "="*45)
print(f"PRÉCISION DE L'ARBRE : {accuracy:.2%}")
print("="*45)

# 6. SAUVEGARDE DU MODÈLE
nom_fichier = "modele_arbre_decision.pkl"
with open(nom_fichier, 'wb') as f:
    pickle.dump(dt_tuned, f)

print(f"\n✅ Modèle sauvegardé avec succès sous le nom : {nom_fichier}")

# Optionnel : Affichage des premières règles de décision
print("\nLogique interne du modèle (Top 2 niveaux) :")
print(dt_tuned.pseudocode(depth=2))
