import os
import re

# 1. Charger le dictionnaire depuis un fichier .txt
def charger_dictionnaire(chemin_dictionnaire):
    with open(chemin_dictionnaire, 'r', encoding='utf-8') as f:
        return [ligne.strip() for ligne in f if ligne.strip()]  # Supprime les espaces et lignes vides

# Remplacez par le chemin de votre fichier dictionnaire.txt
chemin_dictionnaire = "./dictionnaire_medicaux_300.txt"
dictionnaire = charger_dictionnaire(chemin_dictionnaire)

# 2. Dossier à analyser (remplacez par votre chemin)
dossier = "./data"  # Dossier contenant les fichiers à analyser

# 3. Fonction pour générer le vecteur binaire d'un fichier
def generer_vecteur(texte, dictionnaire):
    return [1 if re.search(rf"\b{re.escape(mot)}\b", texte, re.IGNORECASE) else 0
            for mot in dictionnaire]

# 4. Parcourir les fichiers et créer les vecteurs
vecteurs_par_fichier = {}

for fichier in os.listdir(dossier):
    chemin_fichier = os.path.join(dossier, fichier)
    if os.path.isfile(chemin_fichier):
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                texte = f.read()
                vecteur = generer_vecteur(texte, dictionnaire)
                vecteurs_par_fichier[fichier] = vecteur
        except UnicodeDecodeError:
            print(f"⚠️ Le fichier {fichier} n'est pas lisible (UTF-8).")
        except Exception as e:
            print(f"⚠️ Erreur avec {fichier} : {e}")

# 5. Afficher les vecteurs
for fichier, vecteur in vecteurs_par_fichier.items():
    print(f"📄 {fichier} → {vecteur}")