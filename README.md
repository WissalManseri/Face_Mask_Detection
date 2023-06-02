# Détection de masque facial

Ce projet implémente une solution de détection de masque facial en utilisant l'apprentissage en profondeur (deep learning). Le modèle est entraîné à distinguer les images de visages avec et sans masque, en utilisant un réseau de neurones convolutifs (CNN).

# Données
L'ensemble de données utilisé pour l'entraînement et les tests est disponible publiquement sur Kaggle.
Il contient des images de personnes portant et ne portant pas de masques faciaux. Les images sont divisées en deux répertoires, un pour les images avec masque ("with_mask") et un pour les images sans masque ("without_mask").

Environnement
Ce projet utilise Python 3 et plusieurs bibliothèques d'apprentissage automatique en profondeur, y compris TensorFlow et Keras. Vous pouvez installer toutes les bibliothèques nécessaires en exécutant la commande suivante dans votre terminal :

        pip install -r requirements.txt

# Entraînement
Pour entraîner le modèle, vous devez exécuter le script train.py. Avant de l'exécuter, assurez-vous d'avoir correctement configuré les chemins d'accès aux données d'entraînement et de test dans le fichier. Vous pouvez également ajuster les hyperparamètres tels que le taux d'apprentissage, le nombre d'époques, etc.

# Évaluation
Après avoir entraîné le modèle, vous pouvez évaluer ses performances en utilisant le script evaluate.py. Ce script prendra le modèle entraîné et calculera la précision et la perte sur l'ensemble de test.

# Prédiction
Enfin, vous pouvez utiliser le modèle entraîné pour prédire si une image contient ou non un visage avec un masque en utilisant le script predict.py. Ce script prend une image en entrée et renvoie une prédiction pour cette image.

# Conclusion
C'est tout ! Vous pouvez maintenant utiliser ce projet comme point de départ pour entraîner votre propre modèle de détection de masque facial en utilisant vos propres données.
