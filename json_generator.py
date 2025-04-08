import json
import random

def generate_multi_batch_data(num_ids=200, max_batch_num=10, embedding_dim=771):
    """
    Génère une liste de dictionnaires où chaque ID a une entrée pour chaque batch.
    """
    data = []
    global_record_counter = 0 # Pour suivre le nombre total d'enregistrements

    # Boucle sur chaque ID de 1 à num_ids
    for id_val in range(1, num_ids + 1):
        random_number_of_batch = random.randint(1, max_batch_num)
        # Pour chaque ID, boucle sur chaque numéro de batch de 1 à max_batch_num
        for batch_val in range(1, random_number_of_batch + 1):
            target_val = random.randint(0, 1)  # Entier 0 ou 1

            # Génère un vecteur d'embedding unique pour cette combinaison id/batch
            embedding_vec = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]

            record = {
                "id": id_val,          # L'ID (peut être répété)
                "batch": batch_val,    # Le numéro de batch pour cet ID
                "target": target_val,
                "embedding": embedding_vec
            }
            data.append(record)
            global_record_counter += 1

    print(f"Généré {global_record_counter} enregistrements au total ({num_ids} IDs x {max_batch_num} batches/ID).")
    return data

def save_to_json(data_list, filename="donnees_multi_batch.json"):
    """
    Sauvegarde la liste de dictionnaires dans un fichier JSON.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Utilise indent=4 pour une meilleure lisibilité
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"Fichier '{filename}' généré avec succès contenant {len(data_list)} enregistrements.")
    except IOError as e:
        print(f"Erreur lors de l'écriture du fichier '{filename}': {e}")

if __name__ == "__main__":
    nombre_ids_uniques = 200
    nombre_max_batches_par_id = 10 # Chaque ID aura des entrées pour les batches 1 à 10
    taille_embedding = 771
    nom_fichier_sortie = "data.json"

    print(f"Génération de données pour {nombre_ids_uniques} IDs, chacun avec {nombre_max_batches_par_id} batches...")
    donnees_generees = generate_multi_batch_data(
        num_ids=nombre_ids_uniques,
        max_batch_num=nombre_max_batches_par_id,
        embedding_dim=taille_embedding
    )

    print(f"Sauvegarde dans le fichier '{nom_fichier_sortie}'...")
    save_to_json(donnees_generees, nom_fichier_sortie)
