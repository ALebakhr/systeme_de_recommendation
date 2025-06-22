# Système de Recommandation de Films

Ce projet présente une étude approfondie sur la conception et l'évaluation de différents algorithmes de recommandation de films. L'objectif est de comparer plusieurs approches de filtrage collaboratif et de factorisation de matrices, ainsi que des méthodes basées sur la popularité et la similarité entre items.

## Aperçu du projet

Le projet est contenu dans un notebook Jupyter (`movie_recommender_project.ipynb`) qui est structuré comme suit :
- **Chargement et exploration des données** : présentation des jeux de données et analyse descriptive.
- **Préparation des données** : nettoyage, fusion et préparation pour l'entraînement des modèles.
- **Implémentation des algorithmes de recommandation** : filtrage collaboratif (KNN utilisateur et item), factorisation de matrices (SVD, NMF), popularité, et filtrage collaboratif item-item (IBCF).
- **Évaluation et comparaison des performances** : analyse quantitative des résultats.
- **Visualisation et analyse** : illustrations graphiques des distributions et performances.
- **Exemples de recommandations** : génération de recommandations personnalisées pour un utilisateur donné.

## Algorithmes implémentés

Plusieurs algorithmes de recommandation ont été implémentés et évalués :
- **Filtrage Collaboratif Basé sur les Voisins (KNN)**:
    - User-based
    - Item-based
- **Factorisation de Matrices**:
    - SVD (Singular Value Decomposition)
    - NMF (Non-negative Matrix Factorization)
- **Recommandation Basée sur la Popularité**
- **Filtrage Collaboratif Item-Item (IBCF)**:
    - Avec et sans normalisation (centrage utilisateur)

## Données

Le projet utilise les jeux de données suivants:
- `movies.csv`: Contient les informations sur les films (ID, titre, genres).
- `ratings.csv`: Contient les notes données par les utilisateurs aux films (userId, movieId, rating).
- `users.csv`: Contient les informations sur les utilisateurs (ID, genre, occupation).

## Comment utiliser

1.  **Cloner le dépôt.**

2.  **Installer les dépendances :**
    Assurez-vous d'avoir Python 3 installé. Installez les bibliothèques nécessaires via pip en utilisant le fichier `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Si `requirements.txt` n'est pas disponible ou complet, vous pouvez installer les paquets manuellement :
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn surprise jupyter
    ```

3.  **Lancer le notebook Jupyter :**
    ```bash
    jupyter notebook movie_recommender_project.ipynb
    ```
    Vous pouvez ensuite exécuter les cellules du notebook pour voir l'analyse, l'entraînement des modèles et les résultats.

## Évaluation des performances

Les modèles ont été évalués en utilisant les métriques RMSE (Root Mean Squared Error) et MAE (Mean Absolute Error). Les résultats sont les suivants :

| Modèle          | RMSE     | MAE      |
|-----------------|----------|----------|
| KNN utilisateur | 0.963    | 0.763    |
| KNN item        | 0.932    | 0.735    |
| SVD             | 0.874    | 0.686    |
| NMF             | 0.915    | 0.722    |

Les approches basées sur la factorisation de matrices (SVD en particulier) offrent les meilleures performances prédictives pour ce jeu de données.

## Application Web (recommend_web.py)

En plus du notebook, ce projet inclut une application web simple construite avec Flask. Cette application permet aux utilisateurs d'obtenir des recommandations de films de manière interactive en fonction de plusieurs critères.

### Fonctionnalités

- **Filtrage par genre** : Sélectionnez un ou plusieurs genres pour affiner les recommandations.
- **Recherche par mots-clés** : Trouvez des films dont le titre contient des mots-clés spécifiques.
- **Filtrage par année** : Spécifiez une plage d'années de sortie.
- **Filtrage par note** : Définissez une note moyenne minimale.
- **Filtrage par popularité** : Exigez un nombre minimum de notes.
- **Affichage personnalisable** : Choisissez le nombre de recommandations à afficher.

### Comment lancer l'application web

1.  **Assurez-vous que les dépendances sont installées**, y compris Flask:
    ```bash
    pip install Flask pandas surprise
    ```

2.  **Exécutez le script `recommend_web.py`** depuis votre terminal :
    ```bash
    python recommend_web.py
    ```

3.  **Ouvrez votre navigateur** et accédez à l'adresse `http://127.0.0.1:5000` pour utiliser l'application.