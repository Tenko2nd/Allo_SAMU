import docx
import re
import spacy

# Ouvre le document
docsPath = r"Test\Ressources"
record_ID = "test.docx"
doc = docx.Document(docsPath+record_ID)

# Ouvre le modèle de SpaCy
nlp = spacy.load("fr_core_news_lg")

# Ouvre les stopwords et caractères spéciaux
stopwordsPath = r"C:\Projet_synthèse\Ressources\stopwords-fr.txt"
with open(stopwordsPath, 'r', encoding='utf-8') as f:
    stopwords = f.read()
caracterePath = r"C:\Projet_synthèse\Ressources\caracteres_speciaux.txt"
with open(caracterePath, 'r', encoding='utf-8') as f:
    caracteres = f.read()

# met le texte du document dans une liste
allText = [docpara.text for docpara in doc.paragraphs]
allText[0] = " ".join(allText[0].split('\n')[3:])


def clean_and_lemmatize(text, stopwords):
    """
    reformate le texte en token et les lemmatize
    :param text:
    :param stopwords:
    :return:
    """
    # conversion en minuscules et retire les ponctuations
    text = text.lower()
    text = "".join([char if char not in caracteres else ' ' for char in text])

    # Suppression des stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    filtered_words = ' '.join(filtered_words)

    # prépare les mots pour la lemmatization
    document = nlp(filtered_words)
    lemmas = [token.lemma_ for token in document]

    # Filtrage des lemmes vides ou contenant uniquement des espaces
    filtered_lemmas = [lemma for lemma in lemmas if lemma.strip()]

    return filtered_lemmas


# Remplis la liste des mots finaux
cleaned_lemmas = []
for paragraph in allText:
    cleaned_paragraph_lemmas = clean_and_lemmatize(paragraph, stopwords)
    cleaned_lemmas.extend(cleaned_paragraph_lemmas)

print(cleaned_lemmas)



