#!/usr/bin/env python3
"""
filter_word2vec_bin.py

Filtre un fichier d'embeddings word2vec au format binaire (type frWac /
fastText-style "word2vec C binary format") pour ne garder que les vecteurs
correspondant à de vrais mots français (formes fléchies : masculin, féminin,
singulier, pluriel, conjugaisons verbales, etc.), en s'appuyant sur une
liste de mots issue de Lexique383 (lexique.org).

Le fichier est lu et écrit en STREAMING : il n'est jamais chargé entièrement
en mémoire, ce qui permet de traiter de gros fichiers (2 Go et plus) avec
une faible empreinte RAM.

Format binaire attendu (standard word2vec) :
    <nb_mots> <dimension>\n
    puis, pour chaque mot :
        <mot><espace><dim x 4 octets (float32)><\n>

Usage :
    python filter_word2vec_bin.py input.bin output.bin wordlist_fr.txt
"""

import sys
import struct
import argparse


def load_wordlist(path):
    """Charge la liste de mots valides (un mot par ligne) dans un set."""
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words


# Petits connecteurs/élisions qu'on autorise même s'ils ne sont pas des
# entrées lexicales à part entière (utiles pour les tokens avec apostrophe
# ou les "phrases" jointes par underscore, ex: fichier *_phrase_*.bin)
ELISIONS = {"l", "d", "n", "j", "s", "c", "m", "t", "qu", "jusqu", "lorsqu",
            "puisqu", "quoiqu"}


def is_valid_token(token, wordlist, keep_special_tokens=True,
                    keep_phrases=True, min_component_ratio=1.0):
    """
    Détermine si un token (mot du vocabulaire d'embeddings) doit être
    conservé.
    """
    if not token:
        return False

    # Token spécial utilisé par word2vec (padding de fin de phrase)
    if token == "</s>":
        return keep_special_tokens

    low = token.lower()

    # Mot "simple" présent dans le lexique
    if low in wordlist:
        return True

    # Mots avec apostrophe (élision) : l'homme -> "l'" + "homme"
    if "'" in low:
        parts = low.split("'")
        if all(p in wordlist or p in ELISIONS for p in parts if p):
            return True

    # Mots composés avec trait d'union : "porte-monnaie"
    if "-" in low:
        parts = low.split("-")
        valid = [p for p in parts if p in wordlist]
        if parts and len(valid) / len(parts) >= min_component_ratio:
            return True

    # "Phrases" détectées (fichiers *_phrase_*.bin), jointes par "_" :
    # ex. "pomme_de_terre"
    if keep_phrases and "_" in low:
        parts = low.split("_")
        valid = [p for p in parts if p in wordlist or p in ELISIONS]
        if parts and len(valid) / len(parts) >= min_component_ratio:
            return True

    return False


def filter_word2vec_binary(input_path, output_path, wordlist,
                            keep_special_tokens=True, keep_phrases=True,
                            min_component_ratio=1.0, progress_every=200_000):
    """
    Lit input_path (format binaire word2vec), ne garde que les mots
    valides selon `wordlist`, et écrit le résultat dans output_path
    dans le même format binaire.
    """
    with open(input_path, "rb") as fin:
        # --- Lecture de l'entête "nb_mots dim\n" ---
        header = b""
        while True:
            byte = fin.read(1)
            if not byte:
                raise ValueError("Fichier vide ou entête introuvable.")
            if byte == b"\n":
                break
            header += byte
        vocab_size_str, dim_str = header.split()
        vocab_size = int(vocab_size_str)
        dim = int(dim_str)
        vector_bytes = dim * 4  # float32

        print(f"Entête lue : {vocab_size} mots, dimension {dim}")

        kept = 0
        total = 0

        # On écrit dans un fichier temporaire d'abord car on ne connait
        # le nombre de mots gardés qu'à la fin (nécessaire pour l'entête)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "wb") as ftmp:
            for i in range(vocab_size):
                # --- Lecture du mot (jusqu'à l'espace) ---
                word_bytes = bytearray()
                while True:
                    ch = fin.read(1)
                    if not ch:
                        raise ValueError(
                            f"Fin de fichier inattendue à l'entrée {i}"
                        )
                    if ch == b" ":
                        break
                    word_bytes += ch

                # --- Lecture du vecteur ---
                vec = fin.read(vector_bytes)
                if len(vec) != vector_bytes:
                    raise ValueError(
                        f"Vecteur tronqué à l'entrée {i}"
                    )

                # --- Lecture du séparateur '\n' qui suit le vecteur ---
                sep = fin.read(1)
                if sep != b"\n":
                    # certains fichiers n'ont pas toujours ce séparateur ;
                    # on remet le pointeur en arrière si ce n'est pas '\n'
                    fin.seek(-1, 1)

                total += 1

                try:
                    word = word_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    word = word_bytes.decode("latin-1", errors="ignore")

                if is_valid_token(word, wordlist,
                                   keep_special_tokens=keep_special_tokens,
                                   keep_phrases=keep_phrases,
                                   min_component_ratio=min_component_ratio):
                    ftmp.write(word_bytes)
                    ftmp.write(b" ")
                    ftmp.write(vec)
                    ftmp.write(b"\n")
                    kept += 1

                if progress_every and total % progress_every == 0:
                    print(f"  ... {total}/{vocab_size} traités, "
                          f"{kept} conservés")

        # --- Écriture du fichier final avec la bonne entête ---
        with open(output_path, "wb") as fout, open(tmp_path, "rb") as ftmp:
            fout.write(f"{kept} {dim}\n".encode("utf-8"))
            while True:
                chunk = ftmp.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)

        import os
        os.remove(tmp_path)

        print(f"Terminé : {kept}/{total} mots conservés "
              f"({kept/total*100:.1f}%).")

        return total, kept


def main():
    parser = argparse.ArgumentParser(
        description="Filtre un fichier word2vec binaire pour ne garder "
                    "que les vrais mots français."
    )
    parser.add_argument("input", help="Fichier .bin d'entrée")
    parser.add_argument("output", help="Fichier .bin de sortie (filtré)")
    parser.add_argument("wordlist",
                         help="Fichier texte listant les mots valides "
                              "(un mot par ligne), ex: lexique_fr_wordlist.txt")
    parser.add_argument("--drop-special-token", action="store_true",
                         help="Supprime aussi le token spécial </s>")
    parser.add_argument("--no-phrases", action="store_true",
                         help="Ne pas essayer de valider les tokens "
                              "composés avec '_' (phrases)")
    parser.add_argument("--strict-components", action="store_true",
                         help="Exige que TOUS les composants d'un mot "
                              "composé/d'une phrase soient valides "
                              "(par défaut: déjà le cas, ratio=1.0)")
    args = parser.parse_args()

    print("Chargement de la liste de mots français...")
    wordlist = load_wordlist(args.wordlist)
    print(f"{len(wordlist)} mots chargés depuis {args.wordlist}")

    filter_word2vec_binary(
        args.input,
        args.output,
        wordlist,
        keep_special_tokens=not args.drop_special_token,
        keep_phrases=not args.no_phrases,
        min_component_ratio=1.0,
    )


if __name__ == "__main__":
    main()
