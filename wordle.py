from collections import Counter
from copy import copy
import numpy as np
import sys
import tqdm

LEN_WORD = 5

def score_word(word, used_letters, letter_counts, letter_dists, slot_dists, valid_words, validity_emphasis):
    explore_score = 0
    for i in range(LEN_WORD):
        ci = word[i]
        letter_unused = ci not in used_letters
        if letter_unused:
            explore_score += (letter_dists[ci] + slot_dists[i][ci]) / letter_counts[word][ci]
    validity_score = 1 if word in valid_words else 0
    return validity_emphasis * validity_score + (1 - validity_emphasis) * explore_score


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

def get_valid_words(word, feedback, valid_words):
    """
    Given the current guess and feedback from that guess, filter the set
    of valid words to those that are still in play.
    """
    yg_ctr = Counter()
    y_removals = []
    g_removals = []
    x_total_removals = []
    x_spot_removals = []
    # record what we'll need to eliminate in a single pass below
    for i in range(LEN_WORD):
        if feedback[i] == "y":
            y_removals.append((word[i], i))
            yg_ctr[word[i]] += 1
        elif feedback[i] == "g":
            g_removals.append((word[i], i))
            yg_ctr[word[i]] += 1
    for i in range(LEN_WORD):
        if feedback[i] == "x":
            if yg_ctr[word[i]] == 0:
                x_total_removals.append(word[i])
            else:
                x_spot_removals.append((word[i], i))

    # for each word, check against each exclusion, stopping early
    to_remove = set()
    for j, cand_word in enumerate(valid_words):
        keep = True
        for c, i in g_removals:
            # eliminate words that *do not* have the green letter in position i
            if cand_word[i] != c:
                keep = False
                to_remove.add(cand_word)
                break
        if not keep:
            continue
        for c in x_total_removals:
            # if ith letter is neither yellow nor green at another position, eliminate
            # all words containing that letter
            if c in cand_word:
                keep = False
                to_remove.add(cand_word)
                break
        if not keep:
            continue
        for c, i in x_spot_removals:
            # if the ith letter is yellow or green at another position, eliminate
            # words that have the character at position i
            if cand_word[i] == c:
                keep = False
                to_remove.add(cand_word)
                break
        if not keep:
            continue
        for c, i in y_removals:
            # eliminate words that have the yellow letter in position i
            if c not in cand_word or c == cand_word[i]:
                keep = False
                to_remove.add(cand_word)
                break
        if not keep:
            continue
        for c in yg_ctr:
            # for each letter l that is yellow or green at least once, eliminate all words that
            # have fewer than yellow/green_count(l) of ls
            if cand_word.count(c) < yg_ctr[c]:
                to_remove.add(cand_word)
    return valid_words - to_remove

def choose_next_word(used_letters, letter_counts, letter_dists, slot_dists, all_words, valid_words, validity_emphasis):
    """
    Choose the next word by selecting the word that has the highest combined positional score
    """
    scores = np.array([
        score_word(
            w, used_letters, letter_counts,
            letter_dists, slot_dists, valid_words,
            validity_emphasis
        )
        for w in all_words
    ])
    hi_score = np.max(scores)
    idxs = np.argwhere(scores == hi_score).T[0]
    # break ties randomly if necessary
    return np.random.choice(all_words[idxs])

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

def solve(all_words, valid_words, target, max_guesses, init_guess=None):
    """
    Solves a Wordle game given a universe of possible words, a target word, a number
    of guesses, and an (optional) initial guess.
    """
    slot_dists = compute_slot_dists(all_words)
    letter_dists = compute_letter_dists(all_words)
    letter_counts = compute_letter_counts(all_words)
    validity_emphasis = 0
    curr_word = init_guess if init_guess else choose_next_word(
        set(), letter_counts, letter_dists, slot_dists,
        all_words, valid_words, 0
    )
    feedbacks = []
    is_valid = [curr_word in valid_words]
    path = [curr_word]
    used_letters = set(curr_word)
    guesses = 1
    while guesses < max_guesses and curr_word != target:
        # get feedback for current word
        feedback = get_feedback(curr_word, target)
        # eliminate invalid words based on feedback
        valid_words = get_valid_words(curr_word, feedback, valid_words)
        # choose the next word based on the current word, feedback, and remaining valid words
        curr_word = choose_next_word(
            used_letters,
            letter_counts,
            letter_dists,
            slot_dists,
            all_words,
            valid_words,
            validity_emphasis
        )

        feedbacks.append(''.join(feedback))
        path.append(curr_word)
        validity_emphasis += 1 / max_guesses
        used_letters |= set(curr_word)
        is_valid.append(curr_word in valid_words)
        guesses += 1
    return {
        "guesses": path,
        "feedback": feedbacks,
        "valid_words_left": len(valid_words),
        "is_valid": is_valid,
    }

def evaluate(all_words, n_words, n_steps, init_guess=None):
    """
    Evaluate the solver on n_words target words with n_steps guesses allowed
    per solve. Can also provide an optional initial guess if you want to
    evaluate a particular initial guess.
    """
    wins = 0
    valid_words_left = 0
    total_path_length = 0
    valid_words = set(all_words)
    for w in tqdm.tqdm(np.random.choice(all_words, size=n_words, replace=False)):
        sol = solve(all_words, valid_words, w, n_steps, init_guess=init_guess)
        if sol["guesses"][-1] == w:
            wins += 1
            total_path_length += len(sol["guesses"])
        else:
            valid_words_left += sol["valid_words_left"]
    print("win fraction:", wins / n_words)
    print("guesses/win:", total_path_length / wins)
    print("valid words left/loss:", valid_words_left / (n_words - wins) if wins < n_words else 0)

if __name__ == "__main__":
    with open("wordle_words.txt") as wds:
        stripped = [w.strip() for w in wds.readlines()]
        words = np.array([w.lower() for w in stripped if len(w) == 5])
        valid_words = set(words)
    target_word = sys.argv[1] if len(sys.argv) >= 2 else np.random.choice(words)
    if len(sys.argv) < 2:
        print("randomly chosen target word:", target_word)
    else:
        print("target word:", target_word)
    assert target_word in words
    d = solve(words, valid_words, target_word, 6)
    print("guess sequence:", d["guesses"])
    print("feedback sequence:", d["feedback"])
    print("valid words left:", d["valid_words_left"])
    print("is valid", d["is_valid"])

