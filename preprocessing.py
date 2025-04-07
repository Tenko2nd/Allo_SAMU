import os
import docx
import spacy


def docx_to_list(record_ID, record_dir):
    doc = docx.Document(record_dir + record_ID)

    # Ouvre les stopwords et caractères spéciaux
    stopwordsPath = r"Ressources\stopwords-fr.txt"
    with open(stopwordsPath, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    caracterePath = r"Ressources\caracteres_speciaux.txt"
    with open(caracterePath, 'r', encoding='utf-8') as f:
        caracteres = set(line.strip() for line in f)

    # met le texte du document dans une liste
    allText = [docpara.text for docpara in doc.paragraphs]

    cleaned_allText = []
    for paragraph_text in allText:
        cleaned_text = paragraph_text.replace("1-", "")
        cleaned_allText.append(cleaned_text)
    allText = cleaned_allText

    allText[0] = " ".join(allText[0].split('\n')[3:])

    return stopwords, caracteres, allText


def clean_and_lemmatize(text, caracteres, stopwords, nlp):
    """
    reformate le texte en token et les lemmatize
    :param text:
    :param caracteres:
    :param stopwords:
    :param nlp_model: Le modèle SpaCy pré-chargé
    :return:
    """

    text = text.lower()
    text = "".join([char if char not in caracteres else ' ' for char in text])

    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    filtered_words = ' '.join(filtered_words)

    document = nlp(filtered_words)
    lemmas = [token.lemma_ for token in document]

    filtered_lemmas = [lemma for lemma in lemmas if lemma.strip()]

    return filtered_lemmas


def final_words(all_text, caracteres, stopwords, nlp, filename):
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)
    cleaned_lemmas = []
    for paragraph in all_text:
        cleaned_paragraph_lemmas = clean_and_lemmatize(paragraph, caracteres, stopwords, nlp)
        cleaned_lemmas.extend(cleaned_paragraph_lemmas)
    filename_txt = filename.replace(".docx", ".txt")
    filename_txt = os.path.join(data_path, filename_txt)

    with open(filename_txt, 'w', encoding='utf-8') as outfile:
        is_first_lemma = True
        for lemma in cleaned_lemmas:
            if lemma == "speaker" and not is_first_lemma:
                outfile.write("\n")
            outfile.write(lemma + " ")
            is_first_lemma = False

    """
    Dans le dossier "data_bert" on trouvera tous les enregistrements .txt (ayant le même nom que les fichiers d'origine .docx),
    en plus simplifié (uniquement en minuscule, sans l'en-tête, + lemmatisation + suppression des stopwords et de la ponctuation).
    """

    return cleaned_lemmas


if __name__ == "__main__":
    record_dir = r'C:\Users\maeva\Document\ESEO\E4\S2\Projet_synthese\record/' # A modifier avec le dossier où se trouvent les enregistrements .docx
    nlp = spacy.load("fr_core_news_lg")

    filenames = os.listdir(record_dir)

    for filename in filenames:
        if filename.endswith(".docx"):
            record_ID = filename

            stopwords, caracteres, all_text = docx_to_list(record_ID, record_dir)
            final_word_list = final_words(all_text, caracteres, stopwords, nlp, filename)

            print(final_word_list)
            print("-" * 50)
