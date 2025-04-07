from transformers import AutoTokenizer

# Ouvre le document
with open(r"C:\Projet_synthèse\Ressources\test.txt", "r", encoding='utf8') as file:
    text = file.read()

sep = " /"

tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")  # Pas besoin de sep_token ici
tokenizer.add_special_tokens({'sep_token': sep}) # Ajoute / comme token spécial

list_text = text.split(sep)

list_part = []
current_batch = []  # Initialiser le premier batch
previous_turns = []  # Pour stocker les deux derniers tours de parole

for i, text in enumerate(list_text):
    tokens = tokenizer.tokenize(text)  # Tokeniser la phrase courante

    if i >= 2: # On commence à avoir un contexte à partir du 3ème tour (index 2)
        context_tokens = tokenizer.tokenize(sep.join(previous_turns))
    else:
        context_tokens = []

    if len(current_batch) + len(tokens) + len(context_tokens) + (i > 0)  <= 512: # +1 si pas le premier segment
        if i > 0: # Si ce n'est *pas* le premier segment, ajouter le séparateur
          current_batch.append(tokenizer.sep_token) # Ajout correct du sep_token
        current_batch.extend(tokens)
        #current_batch.append(tokenizer.sep_token) # Ajout du sep APRÈS le texte, pas avant le contexte

    else:
        # Le batch actuel est plein
        padding_length = 512 - len(current_batch)
        current_batch.extend([tokenizer.pad_token] * padding_length)
        list_part.append(current_batch)
        # Nouveau batch: contexte + tokens courants
        current_batch = context_tokens # Commence par le contexte
        if context_tokens: # Ajouter un separateur apres le context si il y en a un
            current_batch.append(tokenizer.sep_token)
        current_batch.extend(tokens)
        #current_batch.append(tokenizer.sep_token) #Pas besoin du sep a la fin, on l'aura au debut du suivant

    # Mettre à jour les deux derniers tours de parole
    previous_turns.append(text)
    if len(previous_turns) > 2:
        previous_turns.pop(0)  # Garder seulement les 2 derniers


# Ajouter le dernier batch s'il n'est pas vide
if current_batch:
    padding_length = 512 - len(current_batch)
    current_batch.extend([tokenizer.pad_token] * padding_length)
    list_part.append(current_batch)


# Afficher les batchs (facultatif, pour vérification)
for i, batch in enumerate(list_part):
    print(f"Batch {i + 1}: {len(batch)} tokens")
    print(tokenizer.decode(tokenizer.convert_tokens_to_ids(batch)))  # Utile pour voir le texte

# Convertir en IDs de tokens si nécessaire pour un modèle
list_part_ids = [tokenizer.convert_tokens_to_ids(batch) for batch in list_part]

# Ajout du token de début de phrase (en général [CLS]) et de fin ([SEP])
list_part_ids = [[tokenizer.cls_token_id] + batch + [tokenizer.sep_token_id] for batch in list_part_ids]

for part_ids in list_part_ids:
    print(tokenizer.decode(part_ids))