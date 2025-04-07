import json
import random
import numpy as np


# Fonction pour générer un âge normalisé entre 0 et 1
def generate_normalized_age():
    age = random.randint(30, 120)  # Âge entre 30 et 120 ans
    return (age - 30) / (120 - 30)


# Fonction pour générer un sexe (0 pour femme, 1 pour homme) et répartir les sexes selon les proportions réelles
def generate_sex():
    return random.choice([0, 1])


# Fonction pour générer la dernière valeur du vecteur "embedding"
def generate_last_value():
    return random.choice([0, 0.33, 0.67, 1])


# Fonction pour générer un ID patient aléatoire
def generate_patient_id():
    return random.randint(100000, 999999)


# Liste pour stocker les nouveaux appels simulés
calls_v3 = []

# Génération des 200 appels
for _ in range(200):
    # Générer un embedding avec l'âge, le sexe et la dernière valeur
    embedding = np.random.uniform(-5, 5, 768).tolist()  # Les 768 premières valeurs sont aléatoires entre -5 et 5
    embedding.append(generate_normalized_age())  # Âge normalisé entre 0 et 1
    embedding.append(generate_sex())  # Sexe (0 pour femme, 1 pour homme)
    embedding.append(generate_last_value())  # Dernière valeur aléatoire dans {0, 0.33, 0.67, 1}
    # Créer l'appel
    call = {
        "id": generate_patient_id(),
        "target": random.choice([0, 1]),  # 0 pour non-STEMI, 1 pour STEMI
        "embedding": embedding
    }
    calls_v3.append(call)

# Sauvegarde dans un fichier .json
file_path = 'simulated_calls_v3.json'
with open(file_path, 'w') as f:
    json.dump(calls_v3, f, indent=4)

print(f"Fichier sauvegardé sous {file_path}")