#!/usr/bin/env python3
"""
~~~~~~ Programme fait par Xilly04 assisté par Claude AI ~~~~~~

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
"""




import argparse
import os
import sys
from typing import List, Optional

# Dossier où se trouve ce script, pour retrouver data.bin à côté de lui
# même si le script est lancé depuis un autre dossier.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np

try:
    from gensim.models import KeyedVectors
except ImportError:
    print("gensim n'est pas installé. Lance : pip install gensim numpy")
    sys.exit(1)


# Quelques mots de départ sémantiquement variés, pour explorer large avant
# de resserrer la recherche. Tu peux les changer librement.
SEED_WORDS = [
    "temps", "homme", "femme", "eau", "maison", "animal", "nombre",
    "travail", "amour", "pays", "nature", "corps", "pensée", "musique",
    "nourriture",
]


class CemantixSolver:
    def __init__(self, model_path: str, binary: bool = True, epsilon: float = 0.1):
        print(f"Chargement du modèle depuis {model_path} ...")
        self.kv = KeyedVectors.load_word2vec_format(
            model_path, binary=binary, unicode_errors="ignore"
        )
        print(f"Modèle chargé : {len(self.kv.index_to_key)} mots.")

        # Normalisation des vecteurs une fois pour toutes : la similarité
        # cosinus devient alors un simple produit scalaire, ce qui permet
        # de calculer d'un coup les similarités avec tout le vocabulaire.
        vectors = self.kv.vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.normed_vectors = vectors / norms

        self.word_to_index = {w: i for i, w in enumerate(self.kv.index_to_key)}
        self.epsilon = epsilon

        # Au départ, tous les mots du vocabulaire sont candidats.
        self.candidates = set(self.kv.index_to_key)
        self.tried = set()

    def similarities_with(self, word: str) -> Optional[np.ndarray]:
        """Similarité cosinus entre `word` et tout le vocabulaire (vectorisé)."""
        idx = self.word_to_index.get(word)
        if idx is None:
            return None
        vec = self.normed_vectors[idx]
        return self.normed_vectors @ vec  # produit scalaire = cosinus (vecteurs normés)

    def register_guess(self, word: str, score_pct: float) -> None:
        """
        Enregistre un essai (mot, score affiché par le jeu entre -100 et 100)
        et met à jour l'ensemble des candidats restants.
        """
        word = word.strip().lower()
        self.tried.add(word)
        target_sim = score_pct / 100.0

        sims = self.similarities_with(word)
        if sims is None:
            print(f"⚠️  '{word}' n'est pas dans le vocabulaire local du modèle, "
                  "impossible de l'utiliser comme filtre.")
            return

        new_candidates = set()
        for w in self.candidates:
            i = self.word_to_index[w]
            if abs(sims[i] - target_sim) <= self.epsilon:
                new_candidates.add(w)

        self.candidates = new_candidates - self.tried
        print(f"   -> {len(self.candidates)} candidats restants.")

    def suggest(self, n: int = 10) -> List[str]:
        """
        Propose jusqu'à n mots à essayer ensuite, parmi les candidats
        restants, en privilégiant les mots les plus fréquents (index bas
        dans le vocabulaire gensim = mot fréquent dans le corpus
        d'entraînement).
        """
        remaining = [w for w in self.candidates if w not in self.tried]
        remaining.sort(key=lambda w: self.word_to_index[w])
        return remaining[:n]

    def tighten_epsilon(self, factor: float = 0.7, floor: float = 0.001) -> None:
        """Resserre la marge d'erreur au fil des essais pour affiner la recherche."""
        self.epsilon = max(self.epsilon * factor, floor)


def ask_float(prompt: str) -> Optional[float]:
    raw = input(prompt).strip().replace(",", ".")
    if raw.lower() in ("", "q", "quit", "stop"):
        return None
    try:
        return float(raw)
    except ValueError:
        print("Score invalide, réessaie (ex: 23.45 ou -12.3).")
        return ask_float(prompt)


def main():
    parser = argparse.ArgumentParser(description="Assistant solveur pour Cémantix.")
    parser.add_argument("--model", default=os.path.join(SCRIPT_DIR, "data.bin"),
                         help="Chemin vers le fichier .bin word2vec (défaut: data.bin, dans le même dossier que ce script)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                         help="Marge d'erreur initiale sur la similarité cosinus (défaut: 0.01)")
    parser.add_argument("--suggestions", type=int, default=8,
                         help="Nombre de suggestions affichées à chaque tour")
    args = parser.parse_args()

    solver = CemantixSolver(args.model, epsilon=args.epsilon)

    print("\n=== Assistant Cémantix ===")
    print("Pour chaque mot que tu essaies sur le site, entre ici le mot puis le score affiché.")
    print("Tape 'q' au prompt du score pour quitter.\n")

    print("Suggestions de mots de départ (sémantiquement variés) :")
    print(", ".join(SEED_WORDS))
    print()

    round_num = 1
    while True:
        print(f"--- Tour {round_num} ---")
        word = input("Mot essayé sur Cémantix : ").strip().lower()
        if not word:
            continue
        score = ask_float(f"Score affiché pour '{word}' (ex: 23.45) : ")
        if score is None:
            print("Fin de la session.")
            break

        if score >= 99.99:
            print(f"\n🎉 Trouvé ! Le mot secret est '{word}'.")
            break

        solver.register_guess(word, score)
        solver.tighten_epsilon()

        suggestions = solver.suggest(args.suggestions)
        if suggestions:
            print("Suggestions pour le prochain essai :", ", ".join(suggestions))
        else:
            print("Plus aucun candidat ne correspond : le mot secret n'est "
                  "probablement pas dans ce modèle, ou la marge d'erreur "
                  "(--epsilon) est trop stricte.")
        print()
        round_num += 1


if __name__ == "__main__":
    main()