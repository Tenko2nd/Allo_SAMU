import os
import numpy as np
import pandas as pd
import gensim.downloader as api

# === 1. Charger le modèle FastText (pré-entraîné sur Wikipedia)

print("📦 Chargement du modèle FastText...")
model = api.load("fasttext-wiki-news-subwords-300")  # 300 dimensions
print("✅ Modèle chargé.")

# === 2. Mots-clés médicaux liés à l'infarctus

medical_keywords = {

    "douleur", "poitrine", "mal", "respirer", "essoufflement", "sueur", "nausée", "vertige",

    "palpitation", "fatigue", "angoisse", "ischémie", "obstruction", "rupture", "plaque", "caillot",

    "thrombose", "tabac", "diabète", "hypertension", "cholestérol", "obésité", "obèse", "sédentarité",

    "alcool", "stress", "antécédent", "infarctus", "avc", "cardiaque", "serre", "rétrosternale",

    "gauche", "thoracique", "oppression", "coeur", "hta"

}


# === 3. Fonction pour vectoriser les mots-clés trouvés dans un fichier

def get_embedding_for_keywords(keywords, model):
    vectors = [model[word] for word in keywords if word in model]

    if vectors:
        return np.mean(vectors, axis=0)  # moyenne vectorielle

    return None


# === 4. Parcours des fichiers

data_folder = "data"

results = []

print("📁 Analyse des fichiers...")

for filename in os.listdir(data_folder):

    if filename.endswith(".txt"):

        with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:

            text = f.read().lower()

            lemmas = text.replace("\n", " ").split()

            found_keywords = sorted(set(w for w in lemmas if w in medical_keywords))

            # ✅ Affichage des mots-clés trouvés
            print(f"📝 {filename} → Mots-clés détectés : {found_keywords}")

            embedding = get_embedding_for_keywords(found_keywords, model)

            if embedding is not None:

                row = {"fichier": filename}

                for i in range(len(embedding)):
                    row[f"dim_{i}"] = embedding[i]

                results.append(row)

            else:

                print(f"⚠️ Aucun mot-clé reconnu dans {filename}")

# === 5. Export des résultats

df_embed = pd.DataFrame(results)

df_embed.to_csv("fasttext_embeddings.csv", index=False, encoding='utf-8')

print("✅ Embeddings sauvegardés dans : fasttext_embeddings.csv")

