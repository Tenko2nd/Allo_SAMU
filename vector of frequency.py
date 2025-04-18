import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import OrderedDict

# === 1. Charger le dictionnaire médical ===
def charger_dictionnaire(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        mots = [mot.strip() for mot in f.readlines() if mot.strip()]
        return list(OrderedDict.fromkeys(mots))  # Supprimer les doublons

dictionnaire = charger_dictionnaire("dictionnaire_medicaux_300.txt")

# === 2. Dossier contenant les fichiers texte ===
data_folder = "data"
output_folder = "tfidf_results"
os.makedirs(output_folder, exist_ok=True)

# === 3. Lister les fichiers .txt ===
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]

# === 4. Appliquer le TF-IDF fichier par fichier ===
for filename in txt_files:
    file_path = os.path.join(data_folder, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    docs = [text.strip()]

    # Appliquer le TF-IDF avec dictionnaire médical
    vectorizer = TfidfVectorizer(
        vocabulary=dictionnaire,
        token_pattern=r'(?u)\b\w+\b',
        use_idf=True,
        norm='l2',
        smooth_idf=True
    )
    X_tfidf = vectorizer.fit_transform(docs)

    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Affichage console
    print(f"Document: {filename}")
    print("TF-IDF shape:", X_tfidf.shape)
    print(df_tfidf)
    print("-" * 60)

    # Sauvegarde dans un .txt (lisible)
    out_file = os.path.join(output_folder, filename.replace(".txt", "_tfidf.txt"))
    with open(out_file, 'w', encoding='utf-8') as out:
        out.write(f"Document: {filename}\n")
        out.write(f"TF-IDF shape: {X_tfidf.shape}\n")
        out.write(df_tfidf.to_string(index=False))
