import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Dossier contenant les fichiers texte
data_folder = "data"
output_folder = "tfidf_results"
os.makedirs(output_folder, exist_ok=True)

# Lister les fichiers .txt
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]

for filename in txt_files:
    file_path = os.path.join(data_folder, filename)

    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # On suppose que le texte est déjà nettoyé et lemmatisé
    docs = [text.strip()]  # une seule "phrase" par doc

    # Appliquer TF-IDF
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(docs)

    # Créer un DataFrame pour visualiser
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Affichage console
    print(f"Document: {filename}")
    print("TF-IDF shape:", X_tfidf.shape)
    print(df_tfidf)
    print("-" * 60)

    # Sauvegarder dans un .txt
    out_file = os.path.join(output_folder, filename.replace(".txt", "_tfidf.txt"))
    with open(out_file, 'w', encoding='utf-8') as out:
        out.write(f"Document: {filename}\n")
        out.write(f"TF-IDF shape: {X_tfidf.shape}\n")
        out.write(df_tfidf.to_string(index=False))
