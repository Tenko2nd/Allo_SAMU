## Ordre d'execution des scripts

### 0. Prérequis
Vous devez déposer dans le dossier `Data` les retranscription audio sous format .docx

### 1. Le preprocessing 
Vous avez le choix entre 2 méthodes. Chacune d'elles a besoin de son propre préprocessing.

Le BERT dispose de deux manières de prétraiter les données. Celle qui marche le mieux est le `prepocessing_bert.py`. L'autre donne des résultats moindre avec les modèles testés mais peut rester intéressant pour les tests.

Le TF-IDF propose un seul moyen de prétraitement. Il y a juste à le lancer.

### 2. Vectorization

Pour le BERT, il y a juste à lancer le code `bert_embeddings.py` et il va créer des fichiers JSON dans le sous dossier `Vectoriation/Results`.

Pour le TF-IDF, son implémentation est limité. Il faut lancer dans un premier temps le fichier `metadata.py`, puis le fichier `vector of frequency.py`, et enfin `json_files.py`.
De par ses caractéristiques et son impossibilité d'être simplement intégré dans le projet commun, il risque de poser des problèmes sans revenir dessus et changer manuellement des variables.

### 3. Classification

Pour le BERT, il y a juste à lancer le script `classifier_bert.py` et les résultats sous la forme d'images, csv, etc. avec le nom du model tester. 
On peut retrouver cela dans le sous dossier `Classification/Results/BERT`

Idem pour le TF-IDF.

### Bonus

Pour faire de nouvelles prédictions pour des nouvelles retranscriptions, après avoir fait le préprocessing et la vectorization, on peut utiliser des modèles enregistrés dans le dossier résultats de la classification en lançant le code `prediction_new_case`. 