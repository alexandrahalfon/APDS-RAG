"""Numba JIT-compiled text processing utilities."""

import numpy as np
from numba import jit


@jit(nopython=True)
def find_sentence_boundaries_numba(char_codes: np.ndarray) -> np.ndarray:
    """Find sentence boundary indices in a character code array.

    Sentence boundaries are positions after '.', '!', or '?' followed by
    a space or end-of-text.

    Args:
        char_codes: 1-D int32 array of Unicode code points.

    Returns:
        1-D array of indices where sentences end.
    """
    # Sentence-ending punctuation code points: . ! ?
    dot = 46
    excl = 33
    qmark = 63
    space = 32

    boundaries = np.empty(len(char_codes), dtype=np.int64)
    count = 0

    for i in range(len(char_codes)):
        c = char_codes[i]
        if c == dot or c == excl or c == qmark:
            # Boundary if next char is space or we're at end
            if i == len(char_codes) - 1:
                boundaries[count] = i
                count += 1
            elif char_codes[i + 1] == space:
                boundaries[count] = i
                count += 1

    return boundaries[:count]


@jit(nopython=True)
def count_words_numba(char_codes: np.ndarray) -> int:
    """Count words in a character code array.

    Words are separated by spaces (code point 32).

    Args:
        char_codes: 1-D int32 array of Unicode code points.

    Returns:
        Number of words.
    """
    if len(char_codes) == 0:
        return 0

    space = 32
    word_count = 0
    in_word = False

    for i in range(len(char_codes)):
        if char_codes[i] != space:
            if not in_word:
                word_count += 1
                in_word = True
        else:
            in_word = False

    return word_count
