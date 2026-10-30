# Fichiers à télécharger

Avant de lancer le solveur, il faut télécharger la base de donnée de mots et la trier afin de supprimer les mots incompatible avec le jeu et de réduire l'espace de stockage du fichier.

Cette opération est à lancer une seule fois uniquement.

Base de donnée
--------------
Allez sur le github de Jean Phillipe Fauconnier (https://fauconnier.github.io/) et téléchargez le fichier de 2Gb avec train=cbow, dim=500 et cutoff=10.

Filtre
------
Téléchargez le programme "filtre.py" et "lexique.txt". Placez les dans le même dossier.

Puis, lacez la commande :

    python filtre.py frWac_no_postag_phrase_500_cbow_cut10.bin frWac_filtered.bin lexique.txt

Usage
-----
Placez le fichier "frWac_filtered.bin" filtré dans votre dossier avec "solveur.py" et lancez le programme
