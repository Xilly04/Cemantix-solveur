# Cemantix-solveur

Solveur pour résoudre le jeu Cémantix fait par Xilly04 assisté par Claude AI :



Solveur assistant pour Cémantix (https://cemantix.certitudes.org)

Principe
--------
Cémantix calcule, pour chaque mot proposé, la similarité cosinus entre le
vecteur de ce mot et celui du mot secret, dans un espace d'embeddings
word2vec (modèle frWac de Jean-Philippe Fauconnier, cf. fauconnier.github.io).
Le score affiché ("température") est cette similarité cosinus multipliée
par 100.

Ce script ne triche pas en interrogeant automatiquement l'API du jeu
(dont les conditions d'usage ne sont pas garanties ici). Il fonctionne en
mode "assistant" : tu joues normalement sur le site, tu lui donnes les
mots que tu as essayés et le score que le jeu t'a renvoyé, et il filtre
son propre modèle local pour te proposer le prochain mot le plus
informatif à essayer.

Algorithme
----------
1. On charge le modèle word2vec local (mêmes embeddings que le jeu, en
   théorie -> mêmes similarités).
2. On maintient un ensemble de "candidats" = tout le vocabulaire du
   modèle.
3. À chaque (mot, score) fourni par l'utilisateur :
   - on calcule la similarité cosinus entre ce mot et TOUS les mots du
     vocabulaire local (vectorisé avec numpy pour que ce soit rapide) ;
   - on ne garde comme candidats que les mots dont la similarité avec ce
     mot est proche (± epsilon) du score observé. En effet, le mot
     secret doit lui-même avoir un score proche de ce qu'on a observé
     avec chaque mot déjà essayé.
4. On répète : à chaque nouvel indice, l'ensemble des candidats se
   rétrécit. Le mot proposé est pris parmi les candidats restants les
   plus fréquents (le vocabulaire gensim est trié par fréquence
   décroissante, donc index bas = mot fréquent).
5. Dès qu'un score de 100 (ou très proche) est entré, le mot est trouvé.

Prérequis
---------
    pip install gensim numpy

Usage
-----
    python solveur.py
    python solveur.py --model fichier.bin
