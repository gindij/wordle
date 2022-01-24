from collections import Counter
import pandas as pd
import numpy as np
import sys
from copy import copy
import tqdm

LEN_WORD = 5
USED_LETTER_DISCOUNT = 0.5
VALIDITY_SCORE = 0.25

def get_feedback(word, target):
    """
    Given a word and the target word, get feedback
    """
    feedback = ["x" for _ in range(LEN_WORD)]
    ctr = Counter(target)
    # separate loops because we want to count all greens before counting yellows
    for i in range(LEN_WORD):
        wi = word[i]
        if wi == target[i]:
            feedback[i] = "g"
            ctr[wi] -= 1
    for i in range(LEN_WORD):
        wi = word[i]
        if wi in target and wi != target[i] and ctr[wi] > 0:
            feedback[i] = "y"
            ctr[wi] -= 1
    return feedback


def is_valid_g(gs, word):
    for c, i in gs:
        # eliminate words that *do not* have the green letter in position i
        if word[i] != c:
            return False
    return True

def is_valid_y(ys, word):
    for c, i in ys:
        # eliminate words that have the yellow letter in position i
        if c not in word or c == word[i]:
            return False
    return True

def is_valid_x(xs_spot, xs_total, word):
    for c in xs_total:
        # if ith letter is neither yellow nor green at another position, eliminate
        # all words containing that letter
        if c in word:
            return False
    for c, i in xs_spot:
        # if the ith letter is yellow or green at another position, eliminate
        # words that have the character at position i
        if word[i] == c:
            return False
    return True

def is_valid_yg_counts(yg_counts, word):
    for c in yg_counts:
        # for each letter l that is yellow or green at least once, eliminate all words that
        # have fewer than yellow/green_count(l) of ls
        if word.count(c) < yg_counts[c]:
            return False
    return True

def get_valid_words(word, feedback, valid_words, valid):
    """
    Given the current guess and feedback from that guess, filter the set
    of valid words to those that are still in play.
    """
    yg_ctr = Counter()
    ys = []
    gs = []
    xs_total = []
    xs_spot = []
    # record what we'll need to eliminate in a single pass below
    for i in range(LEN_WORD):
        if feedback[i] == "y":
            ys.append((word[i], i))
            yg_ctr[word[i]] += 1
        elif feedback[i] == "g":
            gs.append((word[i], i))
            yg_ctr[word[i]] += 1
    for i in range(LEN_WORD):
        if feedback[i] == "x":
            if yg_ctr[word[i]] == 0:
                xs_total.append(word[i])
            else:
                xs_spot.append((word[i], i))

    # for each word, check against each exclusion, stopping early
    new_valid_words = copy(valid_words)
    for cand_word in valid_words:
        if not (
            is_valid_x(xs_spot, xs_total, cand_word)
            and is_valid_g(gs, cand_word)
            and is_valid_y(ys, cand_word)
            and is_valid_yg_counts(yg_ctr, cand_word)
        ):
            del new_valid_words[cand_word]
            valid[valid_words[cand_word]] = 0
    return new_valid_words

def choose_next_word(
    used_letters,
    likelihood_scores,
    valid,
    all_words,
    validity_emphasis
):
    """
    Choose the next word by selecting the word that has the highest combined positional score
    """
    def len_common(s):
        ln = 0
        for c in s:
            if c in used_letters:
                ln += 1
        return ln
    reps = np.array([len_common(w) for w in all_words])
    scores = validity_emphasis * valid * VALIDITY_SCORE + (1 - validity_emphasis) * likelihood_scores / (reps + 1)
    return all_words[np.argmax(scores)]

def solve(
    all_words,
    valid_words,
    target,
    max_guesses,
    likelihood_scores,
    init_guess=None
):
    """
    Solves a Wordle game given a universe of possible words, a target word, a number
    of guesses, and an (optional) initial guess.
    """
    valid = np.ones(len(all_words))
    curr_word = init_guess if init_guess else choose_next_word(
        used_letters=set(),
        all_words=all_words,
        valid=valid,
        likelihood_scores=likelihood_scores,
        validity_emphasis=0
    )
    feedbacks = []
    is_valid = [True]
    path = [curr_word]
    used_letters = set(curr_word)
    while len(path) < max_guesses and curr_word != target:
        # get feedback for current word
        feedback = get_feedback(curr_word, target)
        # eliminate invalid words based on feedback
        valid_words = get_valid_words(curr_word, feedback, valid_words, valid)
        # choose the next word based on the current word, feedback, and remaining valid words
        curr_word = choose_next_word(
            used_letters=used_letters,
            all_words=all_words,
            valid=valid,
            likelihood_scores=likelihood_scores,
            validity_emphasis=len(path) / max_guesses
        )
        feedbacks.append(''.join(feedback))
        path.append(curr_word)
        used_letters |= set(curr_word)
        is_valid.append(curr_word in valid_words)
    return {
        "guesses": path,
        "feedback": feedbacks,
        "valid_words": list(valid_words.keys()),
        "valid_words_left": len(valid_words),
        "is_valid": is_valid,
    }

def evaluate(
    all_words,
    valid_words,
    n_words,
    max_guesses,
    likelihood_scores,
    init_guess=None
):
    """
    Evaluate the solver on n_words target words with max_guesses guesses allowed
    per solve. Can also provide an optional initial guess if you want to
    evaluate a particular initial guess.
    """
    wins = 0
    valid_words_left = 0
    total_path_length = 0
    failed_words = set()
    word_sample = tqdm.tqdm(np.random.choice(all_words, size=n_words, replace=False))
    for w in word_sample:
        sol = solve(
            all_words,
            valid_words,
            w,
            max_guesses,
            likelihood_scores,
            init_guess=init_guess
        )
        if sol["guesses"][-1] == w:
            wins += 1
            total_path_length += len(sol["guesses"])
        else:
            valid_words_left += sol["valid_words_left"]
            failed_words.add(w)
    print("win fraction:", wins / n_words)
    print("guesses/win:", total_path_length / wins)
    print("valid words left/loss:", valid_words_left / (n_words - wins) if wins < n_words else 0)
    print("failed initial guesses:", failed_words)

def compute_letter_dists(words):
    """
    Compute distributions of letters across all words.
    """
    all_letters = ''.join(words)
    letter_counts = Counter(all_letters)
    for l in letter_counts:
        letter_counts[l] /= len(all_letters)
    return letter_counts

def compute_slot_dists(words):
    """
    Compute distributions of letters in each (of 5) slots.
    """
    ctrs = [Counter([w[i] for w in words]) for i in range(LEN_WORD)]
    for i in range(LEN_WORD):
        for l in ctrs[i]:
            ctrs[i][l] /= len(words)
    return ctrs

def compute_letter_counts(words):
    """
    Compute the counts of letters in each word.
    """
    return {word: Counter(word) for word in words}

def compute_likelihood_score(
    word,
    letter_dists,
    slot_dists,
    letter_counts,
):
    """
    Compute likelihood scores for each word. Score consists of positional and
    letter frequency scores.
    """
    lik_score = 0
    for i in range(LEN_WORD):
        c = word[i]
        lik_score += (letter_dists[c] + slot_dists[i][c]) / letter_counts[c]
    return lik_score

if __name__ == "__main__":
    with open("wordle_words.txt") as wds:
        stripped = [w.strip() for w in wds.readlines()]
        words = np.array([w.lower() for w in stripped if len(w) == LEN_WORD])
    words = list(set(words))
    valid_words = {words[i]: i for i in range(len(words))}
    slot_dists = compute_slot_dists(words)
    letter_dists = compute_letter_dists(words)
    letter_counts = compute_letter_counts(words)
    likelihood_scores = np.array([compute_likelihood_score(w, letter_dists, slot_dists, letter_counts[w]) for w in words])
    target_word = sys.argv[1] if len(sys.argv) >= 2 else np.random.choice(words)
    if len(sys.argv) < 2:
        print("randomly chosen target word:", target_word)
    else:
        print("target word:", target_word)
    assert target_word in words
    d = solve(words, valid_words, target_word, 6, likelihood_scores)
    print("guess sequence:", d["guesses"])
    print("feedback sequence:", d["feedback"])
    print("valid_words_left:", d["valid_words"])
    print("valid words left:", d["valid_words_left"])
    print("is valid:", d["is_valid"])

    # import cProfile
    # with cProfile.Profile() as prof:
    evaluate(words, valid_words, 100, 6, likelihood_scores)
    # prof.print_stats()
