import pandas as pd

# Liste combinée de 300 mots médicaux
mots_medicaux = [
    # --- 100 premiers mots génériques ---
    "douleur", "poitrine", "mal", "essoufflement", "respirer", "sueur", "suer", "vertige", "fatiguer", "pâle", "fatigue", "angoisse",
    "palpitation", "ischémie", "obstruction", "rupture", "plaque", "caillot", "thrombose", "tabac", "diabète",
    "hypertension", "cholestérol", "obésité", "obèse", "sédentarité", "alcool", "stress", "antécédent", "infarctus",
    "avc", "cardiaque", "serre", "rétrosternale", "gauche", "thoracique", "oppression", "coeur", "hta",
    "crise", "malaise", "sensation", "douleur", "compression", "pression", "chaleur", "picotement", "brûlure",
    "brûlant", "sensation", "détresse", "trouble", "difficulté", "inconfort", "évanouissement", "étourdissement",
    "halètement", "trouble", "infarctus", "troubles", "troubles", "artère", "myocarde", "nécrose", "ischémique",
    "arythmie", "tachycardie", "bradycardie", "asystolie", "bloc", "coronaire", "stent", "angioplastie",
    "électrocardiogramme", "pression", "diastolique", "systolique", "catheter", "thrombus", "dysfonction", "ventricule",
    "ventriculaire", "fibrillation", "myocardite", "pericardite", "cyanose", "œdème", "syncope", "hypoxie", "hypotension",
    "régurgitation", "valvulaire", "dyspnée", "tachypnée", "hypopnée", "hypoxémie", "anoxie", "hémorragie", "bruit",
    "souffle", "échocardiographie", "valvulopathie", "saturation", "vasodilatation", "vasoconstriction", "angiographie"
] + [
    # --- 200 mots supplémentaires ---
    "allergie", "analgésique", "anesthésie", "antibiotique", "antidouleur", "antipyrétique", "antiseptique",
    "apnée", "asphyxie", "asystolie", "bactérie", "battement", "biopsie", "bronche", "bronchite",
    "carcinome", "carotide", "céphalée", "chirurgie", "circulation", "coma", "contusion",
    "coronarien", "coupure", "crâne", "cyanose", "défibrillateur", "démence", "dérèglement", "déshydratation",
    "diarrhée", "dissection", "dorsalgie", "embolie", "endocrine", "épilepsie", "épistaxis", "erythème", "exsudat",
    "fracture", "gastro", "gastro-entérite", "hématome", "hémoptysie", "hémorragie", "hépatite", "hernie",
    "hypercapnie", "hyperthermie", "hypothermie", "hypotension", "ictère", "infection", "inflammation",
    "injection", "insuffisance", "intervention", "intoxication", "intubation", "intraveineuse", "lésion",
    "luxation", "méningite", "migraine", "néoplasie", "nodule", "œsophage", "paralysie", "pathologie",
    "péricardite", "perfusion", "phlébite", "phobie", "pneumonie", "pneumothorax", "ponction", "prothèse",
    "prurit", "psychiatrie", "radiographie", "réanimation", "réflexe", "rhume", "saignement", "scanner",
    "sclérose", "sécrétion", "septicémie", "soin", "sonde", "souffrance", "spasme", "stérile", "suture",
    "tachypnée", "tension", "toux", "toxine", "traitement", "ulcère", "urgence", "urine", "utérus", "valvule",
    "vomir", "anémie", "asepsie", "asthénie", "aorte", "arythmie", "bradyarythmie", "bruit", "brûlure", "cellule",
    "cirrhose", "colique", "colonne", "crampe", "cyanose", "décompensation", "désaturation", "diurèse",
    "électrocardiogramme", "emphysème", "enflure", "enrouement", "érythème", "fécalome", "fistule", "gencive",
    "glucose", "hallucination", "hémiplégie", "hémoptysie", "hémorragie", "hyperplasie", "hypersensibilité",
    "hypoxie", "ictus", "immunodéficience", "inflammation", "intestin", "ischémie", "jambes", "lésion", "lumbago",
    "masse", "médicament", "miction", "névralgie", "occlusion", "œsophagite", "pancréatite", "paresthésie",
    "pathogène", "péricarde", "plaie", "plèvre", "pression", "psoriasis", "récidive", "rhinite", "rhumatisme",
    "sclérose", "spasticité", "tachyarythmie", "tachycardie", "thermomètre", "thrombose", "tumeur", "valvulopathie"
]

# Sauvegarde au format .txt
with open("dictionnaire_medicaux_300.txt", "w", encoding="utf-8") as f:
    for mot in mots_medicaux:
        f.write(mot + "\n")

print("✅ Fichier dictionnaire_medicaux_300.txt généré.")
