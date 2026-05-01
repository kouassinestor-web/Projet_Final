import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# --- CONFIGURATION INITIALE ---
# Téléchargement des ressources nécessaires (indispensable pour le déploiement)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stemmer = SnowballStemmer('french')
stop_words = set(stopwords.words('french'))

# --- FONCTION DE PRÉTRAITEMENT ---
def extraire_features(phrase):
    """Applique le même nettoyage que lors de l'entraînement [cite: 1, 2]"""
    if not isinstance(phrase, str): return {}
    tokens = word_tokenize(phrase.lower(), language='french')
    # On garde les racines (stems) des mots significatifs
    stems = [stemmer.stem(m) for m in tokens if m.isalpha() and m not in stop_words]
    return {mot: True for mot in stems}

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource # Évite de recharger le modèle à chaque clic
def load_model():
    # Remplace par le chemin relatif si le fichier est dans le même dossier
    with open('modele_arbre_decision.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    model = None

# --- INTERFACE UTILISATEUR ---
st.title("🎫 Classification Support Client")
st.markdown("Saisissez un message pour déterminer automatiquement sa catégorie.")

message_input = st.text_area("Message du client :", placeholder="Ex: Mon colis n'est pas arrivé...")

if st.button("Analyser le message"):
    if message_input.strip() != "" and model is not None:
        # 1. Extraction des features
        features = extraire_features(message_input)
        
        # 2. Prédiction
        prediction = model.classify(features)
        
        # 3. Affichage du résultat
        st.success(f"**Catégorie détectée :** {prediction}")
        
        # Petit bonus : affichage des mots clés détectés par le modèle
        st.info(f"Mots-clés extraits : {', '.join(features.keys())}")
    else:
        st.warning("Veuillez entrer un message valide.")

# --- FOOTER ---
st.divider()
st.caption("Modèle : Arbre de Décision (NLTK) | Langue : Français")