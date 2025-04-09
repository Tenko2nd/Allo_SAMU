import os

# Ensemble de tous les mots-clés médicaux liés à l'infarctus

medical_keywords = {"douleur", "poitrine", "mal" , "respirer", "essoufflement", "sueur", "nausée", "vertige", "palpitation", "fatigue",
                    "angoisse", "ischémie", "obstruction", "rupture", "plaque", "caillot", "thrombose", "tabac",
                    "diabète", "hypertension", "cholestérol", "obésité", "obèse", "sédentarité", "alcool", "stress",
                    "antécédent", "infarctus", "avc", "cardiaque", "serre","rétrosternale", "gauche","thoracique", "oppression",
                    "coeur", "hta"}


def detect_keywords(lemmas, keywords):
    return sorted(set(lemma for lemma in lemmas if lemma in keywords))


def analyze_folder(folder_path):
    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().lower()

                lemmas = content.replace("\n", " ").split()

            found_keywords = detect_keywords(lemmas, medical_keywords)

            print(f"\n📄 Fichier : {filename}")

            print("Mots-clés médicaux détectés :",
                  ", ".join(found_keywords) if found_keywords else "Aucun mot-clé détecté.")


# 📁 Remplace ce chemin par ton dossier contenant les fichiers .txt

preprocessed_folder = r"data"

analyze_folder(preprocessed_folder)

