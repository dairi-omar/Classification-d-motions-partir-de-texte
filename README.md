<h1 style="color: red;  text-align: center;">Classification d’émotions à partir de texte</h1>

## 🎯 Objectif

Ce projet vise à construire un modèle de deep learning pour la reconnaissance et la classification des émotions à partir de textes courts, tels que les Tweets et les posts sur les réseaux sociaux.

Le modèle est entraîné à reconnaître six émotions principales :
* Joie (joy)
* Tristesse (sadness)
* Colère (anger)
* Peur (fear)
* Amour (love)
* Surprise (surprise)

## 💾 Dataset

Nous utilisons le "Emotions Dataset for NLP" disponible sur Kaggle. Le notebook utilise la répartition suivante :
* **Entraînement :** 16 000 exemples (`train.txt`)
* **Test / Validation :** 2 000 exemples (`test.txt`)

## 🛠️ Pipeline de Traitement

1.  **Chargement des données :** Les fichiers `.txt` sont chargés dans des DataFrames Pandas.
2.  **Nettoyage du texte :** Une fonction `clean_text` est appliquée pour :
    * Mettre le texte en minuscule.
    * Supprimer tous les caractères non alphabétiques (ponctuation, chiffres) via regex.
    * Supprimer les espaces superflus.
3.  **Encodage des étiquettes :** Les 6 étiquettes textuelles (joie, tristesse, etc.) sont transformées en vecteurs one-hot en utilisant `LabelEncoder` de Scikit-learn et `to_categorical` de Keras.
4.  **Tokenisation et Padding :**
    * Le texte est tokenisé avec le `Tokenizer` de Keras (vocabulaire limité à 10 000 mots).
    * Les phrases sont converties en séquences d'entiers.
    * Les séquences sont normalisées à une longueur fixe de **50 tokens** via `pad_sequences`.

## 🧠 Architecture du Modèle

Le modèle est un `Sequential` de Keras basé sur des couches LSTM bidirectionnelles :

1.  **Embedding** : (input_dim=10000, output_dim=128, input_length=50)
2.  **LSTM(128)** (avec dropout de 0.3)
3.  **LSTM(64)** (avec dropout de 0.3)
4.  **Dense** (64 neurones, activation 'relu')
5.  **Dropout** (0.5)
6.  **Dense** (6 neurones, activation 'softmax') - Couche de sortie pour les 6 classes.

## 🚀 Entraînement et Résultats

* **Optimiseur :** Adam
* **Fonction de perte :** `categorical_crossentropy`
* **Métriques :** Accuracy
* **Époques :** 10
* **Taille de lot (Batch Size) :** 256

Le modèle atteint une **précision (accuracy) de 90.45%** sur l'ensemble de test. Les courbes ROC et les scores AUC pour chaque classe confirment d'excellentes performances (AUC > 0.96 pour toutes les classes).

## ⚙️ Comment l'utiliser

1.  Clonez ce dépôt.
2.  Assurez-vous d'avoir le dataset (`train.txt`, `test.txt`) disponible.
3.  **Important :** Mettez à jour la variable `datasetPath` dans la 5ème cellule de code pour pointer vers l'emplacement de votre dossier `DataSet`. Le notebook utilise un chemin absolu :
    ```python
    datasetPath = r"D:\Projects\EmotionsFromText\DataSet" 
    # (À CHANGER)
    ```
4.  Installez les dépendances requises :
    ```bash
    pip install -r requirements.txt
    ```
5.  Exécutez le notebook `EmotionsFromText.ipynb` (par exemple, avec Jupyter Lab ou VS Code).
