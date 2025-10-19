<h1 style="color: red;  text-align: center;">Classification d’émotions à partir de texte</h1>

## 🎯 Objectif

Ce projet vise à construire et évaluer des modèles de deep learning (RNN) pour la reconnaissance et la classification des émotions à partir de textes courts, tels que les Tweets et les posts sur les réseaux sociaux.

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

1.  **Chargement des données :** Les fichiers `.txt` sont chargés dans des DataFrames Pandas, séparés par `;`.
2.  **Nettoyage du texte :** Une fonction `clean_text` est appliquée pour :
    * Mettre le texte en minuscule.
    * Supprimer tous les caractères non alphabétiques (ponctuation, chiffres) via regex.
    * Supprimer les espaces superflus.
3.  **Encodage des étiquettes :** Les 6 étiquettes textuelles (joie, tristesse, etc.) sont transformées en vecteurs one-hot en utilisant `LabelEncoder` de Scikit-learn et `to_categorical` de Keras.
4.  **Tokenisation et Padding :**
    * Le texte est tokenisé avec le `Tokenizer` de Keras (vocabulaire limité à 10 000 mots).
    * Les phrases sont converties en séquences d'entiers.
    * Les séquences sont normalisées à une longueur fixe de **50 tokens** via `pad_sequences`.

## 🧠 Modèles et Résultats

Deux architectures de réseaux de neurones récurrents ont été testées :

### 1. Modèle LSTM
* **Architecture :** `Embedding(10000, 128)` -> `LSTM(128, return_seq=True)` -> `LSTM(64)` -> `Dense(64, 'relu')` -> `Dropout(0.5)` -> `Dense(6, 'softmax')`.
* **Performance (Test) :** **89.25%** d'accuracy.

### 2. Modèle GRU (Gated Recurrent Unit)
* **Architecture :** `Embedding(10000, 128)` -> `GRU(128, return_seq=True)` -> `GRU(64)` -> `Dense(64, 'relu')` -> `Dropout(0.5)` -> `Dense(6, 'softmax')`.
* **Performance (Test) :** **91.35%** d'accuracy.

### Conclusion
Le modèle **GRU** s'est montré plus performant que le modèle LSTM pour cette tâche de classification, avec une précision de 91.35% sur l'ensemble de test.

## 💾 Modèle Sauvegardé

Le modèle GRU (le plus performant) est sauvegardé dans le fichier `EmotionsFromText_model_gru.h5`.

## ⚙️ Comment l'utiliser

1.  Clonez ce dépôt.
2.  Assurez-vous d'avoir le dataset (`train.txt`, `test.txt`) de Kaggle.
3.  **Important :** Mettez à jour la variable `datasetPath` dans la 5ème cellule de code pour pointer vers l'emplacement de votre dossier `DataSet`. Le notebook utilise un chemin absolu qui doit être modifié :
    ```python
    # Cellule 5
    datasetPath = r"D:\Projects\EmotionsFromText\DataSet" 
    # (À CHANGER)
    ```
4.  Installez les dépendances requises (voir `requirements.txt`) :
    ```bash
    pip install pandas seaborn matplotlib scikit-learn tensorflow
    ```
5.  Exécutez le notebook `EmotionsFromText.ipynb`.
