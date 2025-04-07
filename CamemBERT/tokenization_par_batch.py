from transformers import AutoTokenizer
import camembert_constant as c


def tokenize_text(file_path: str, model_name: str, verification: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(file_path, "r", encoding='utf8') as file:
        text = file.read()

    # Diviser en tours de parole (en gardant le texte de chaque tour)
    # strip() enlève les espaces au début/fin, filter(None, ...) enlève les chaînes vides
    turns_text = list(filter(None, [turn.strip() for turn in text.split(c.TEXT_FILE_SEPARATOR)]))

    if not turns_text:
        print("Erreur : Aucun tour de parole trouvé dans le texte.")
        return None

    # --- Logique de Batching ---
    batched_input_ids = []
    batched_attention_masks = []
    current_turn_index = 0

    while current_turn_index < len(turns_text):
        # 1. Construire le Contexte
        context_start_index = max(0, current_turn_index - c.CONTEXT_LEN)
        context_turns = turns_text[context_start_index: current_turn_index]

        # Ajout du contexte
        context_tokens = []
        if context_turns:
            for i, turn in enumerate(context_turns):
                context_tokens.extend(tokenizer.tokenize(turn))
                if i < len(context_turns) - 1:
                    context_tokens.append(tokenizer.sep_token)

        # 2. Ajouter les tours de parole au batch courant
        batch_current_turns_tokens = []
        turns_added_in_this_batch = 0
        temp_turn_index = current_turn_index

        while temp_turn_index < len(turns_text):
            turn_to_add_text = turns_text[temp_turn_index]
            turn_to_add_tokens = tokenizer.tokenize(turn_to_add_text)

            # Calculer la longueur potentielle du batch SI on ajoute ce tour
            # Tokens = CLS + Contexte + Batch + SEP + Nouveau_Tour + SEP_final
            tokens_in_batch_so_far = len(context_tokens) + len(batch_current_turns_tokens)

            # Nombre de séparateurs (</s>) à ajouter AVANT ce nouveau tour:
            # 1 si contexte ET c'est le premier tour du batch
            # 1 si ce n'est PAS le premier tour du batch
            separators_needed_before_new_turn = 0
            if context_tokens and turns_added_in_this_batch == 0:
                separators_needed_before_new_turn = 1
            elif turns_added_in_this_batch > 0:
                separators_needed_before_new_turn = 1

            # +1 pour CLS (<s>), +1 pour le SEP final (</s>)
            potential_len = 1 + tokens_in_batch_so_far + separators_needed_before_new_turn + len(turn_to_add_tokens) + 1

            if potential_len <= c.MAX_LEN:
                if separators_needed_before_new_turn == 1:
                    batch_current_turns_tokens.append(tokenizer.sep_token)
                batch_current_turns_tokens.extend(turn_to_add_tokens)
                turns_added_in_this_batch += 1
                temp_turn_index += 1
            else:
                # Le tour ne rentre pas, on arrête d'ajouter pour ce batch
                # Gérer le cas où même le premier tour (avec contexte) est trop long
                if turns_added_in_this_batch == 0:
                    print(f"Attention : Le tour {current_turn_index} ('{turn_to_add_text[:50]}...') "
                          f"est trop long ({potential_len} tokens avec contexte) pour tenir dans un batch de {c.MAX_LEN}. "
                          f"Il sera tronqué ou ignoré si l'espace est insuffisant.")

                    # Tentative de troncature (simple) si possible
                    available_space = c.MAX_LEN - (1 + len(context_tokens) + separators_needed_before_new_turn + 1)
                    if available_space > 10:  # Garder une marge minimale
                        if separators_needed_before_new_turn == 1:
                            batch_current_turns_tokens.append(tokenizer.sep_token)
                        batch_current_turns_tokens.extend(turn_to_add_tokens[:available_space])
                        turns_added_in_this_batch += 1
                        temp_turn_index += 1  # On a ajouté (partiellement) ce tour
                        print(f"Tronqué à {available_space} tokens.")
                    else:
                        print("Espace insuffisant même pour tronquer. Ce tour sera sauté pour éviter boucle infinie.")
                        temp_turn_index += 1

                break

        if turns_added_in_this_batch == 0 and current_turn_index < len(turns_text):
            # Cela peut arriver si le premier tour était trop long et a été sauté.
            print(f"Alerte: Aucun tour n'a pu être ajouté au batch démarrant au tour {current_turn_index}. "
                  f"Avancement forcé de l'index à {temp_turn_index}.")
            current_turn_index = temp_turn_index
            continue

        # 3. Finaliser le Batch Courant
        # Assembler les tokens : CLS + Contexte + Tours_Batch + SEP_final
        final_tokens = [tokenizer.cls_token]  # Start with CLS (<s>)
        final_tokens.extend(context_tokens)
        final_tokens.extend(batch_current_turns_tokens)
        final_tokens.append(tokenizer.sep_token)  # End with SEP (</s>)

        # Convertir en IDs
        input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

        # Créer le masque d'attention
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = c.MAX_LEN - len(input_ids)
        if padding_length < 0:
            print(
                f"Erreur: Dépassement de MAX_LEN détecté ({len(input_ids)} > {c.MAX_LEN}) pour le batch commençant au"
                f" tour {current_turn_index}. Tronquature forcée.")
            input_ids = input_ids[:c.MAX_LEN]
            attention_mask = attention_mask[:c.MAX_LEN]
            # S'assurer que le dernier token est bien SEP si on a tronqué
            if tokenizer.convert_tokens_to_ids(tokenizer.sep_token) != input_ids[-1]:
                input_ids[-1] = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            padding_length = 0

        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)

        # Ajouter aux listes finales
        batched_input_ids.append(input_ids)
        batched_attention_masks.append(attention_mask)

        # 4. Mettre à jour l'index pour le prochain batch
        current_turn_index = temp_turn_index

    if verification:
        verification_tokeniztion(batched_input_ids=batched_input_ids,
                                 batched_attention_masks=batched_attention_masks,
                                 tokenizer=tokenizer)

    return batched_input_ids, batched_attention_masks


def verification_tokeniztion(batched_input_ids, batched_attention_masks, tokenizer):
    print(f"\n--- TOTAL BATCHES CREES : {len(batched_input_ids)} ---")
    for i, (ids, mask) in enumerate(zip(batched_input_ids, batched_attention_masks)):
        print(f"\n--- Batch {i + 1} ---")
        print(f"Taille totale (avec padding): {len(ids)}")
        num_real_tokens = sum(mask)
        print(f"Nombre de tokens réels: {num_real_tokens}")
        print(f"Nombre de tokens de padding: {c.MAX_LEN - num_real_tokens}")

        # Décoder pour vérifier le contenu (uniquement les tokens réels)
        decoded_text = tokenizer.decode(ids[:num_real_tokens], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=True)
        print(f"Contenu décodé:\n{decoded_text}\n")

        # Vérifier les tokens spéciaux
        print(f"Commence par CLS ({tokenizer.cls_token}): {tokenizer.decode([ids[0]]) == tokenizer.cls_token}")
        # Le dernier token *réel* doit être SEP
        print(
            f"Finit (avant padding) par SEP ({tokenizer.sep_token}): "
            f"{tokenizer.decode([ids[num_real_tokens - 1]]) == tokenizer.sep_token}")
        if c.MAX_LEN - num_real_tokens > 0:
            print(f"Token de padding ({tokenizer.pad_token}): {tokenizer.decode([ids[-1]]) == tokenizer.pad_token}")
        else:
            print("Pas de padding.")
