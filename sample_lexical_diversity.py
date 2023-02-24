import re
import json
import string
import math
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import OrderedDict
from collections import defaultdict
from random import choice
from typing import List, Set
from copy import deepcopy


stops = set(stopwords.words("english"))
ps = PorterStemmer()

# reorder the list
# now the list is going to be a random shuffle of sorted_l[:top_p fraction] + a random shuffle of sorted_l[top_p fraction:]
def re_order(sorted_l: List[str], top_p: float) -> List[str]:
    part1 = sorted_l[:int(len(sorted_l) * top_p) + 1]
    part2 = sorted_l[int(len(sorted_l) * top_p) + 1:]
    np.random.shuffle(part1)
    np.random.shuffle(part2)
    return part1 + part2


def get_word_set_of_sample(sample: str) -> Set[str]:
    sample_no_punc = sample.translate(str.maketrans('', '', string.punctuation))
    word_set = {ps.stem(word) for word in word_tokenize(sample_no_punc) if word not in stops}
    return word_set


# sorted_A is a sorted list of samples for Corpus A, with more representative samples at the beginning
# sorted_B is a sorted list of samples for Corpus B, with more representative samples at the beginning
# top_p is the fraction of samples we want to focus on, but if we run out of samples, we will use the rest of the samples
# num_samples is the number of samples we want to return for each group
def lexical_diversity(sorted_A: List[str], sorted_B: List[str], top_p: float = 0.2, num_samples: int = 4, max_gap=None):
    sorted_A, sorted_B = deepcopy(sorted_A), deepcopy(sorted_B)
    
    a_candidates = [] 
    b_candidates = []

    if max_gap is None:
        max_gap = (num_samples // 4 + 1)

    reordered_A = re_order(sorted_A, top_p)
    reordered_B = re_order(sorted_B, top_p)

    a_words_count, b_words_count = defaultdict(int), defaultdict(int)

    # enumerate through the samples
    # keeps track of how many samples we have examined
    cur_A_pointer, cur_B_pointer = 0, 0

    # we add 1 sentence for group A and group B alternatively, until we have num_samples samples
    for _ in range(num_samples):
        # add a sentence for group A
        # enumerate the sentence from the reordered_A list until we find a legitimate sentence
        while cur_A_pointer < len(reordered_A):

            # get the sentence
            sample_A = reordered_A[cur_A_pointer]
            cur_A_pointer += 1

            # get the set of words of the sentence
            word_set_A = get_word_set_of_sample(sample_A)

            # decide whether to add the sentence
            add_A_flg = True
            for word in word_set_A:
                if a_words_count[word] - b_words_count[word] >= max_gap:
                    add_A_flg = False
                    break

            # if we decide to add the sentence, add it to the candidate list and then stop the while loop
            if add_A_flg:
                a_candidates.append(sample_A)
                for word in word_set_A:
                    a_words_count[word] += 1
                break
        
        # same as above, but for group B instead of group A
        while cur_B_pointer < len(reordered_B):
            sample_B = reordered_B[cur_B_pointer]
            cur_B_pointer += 1
            word_set_B = get_word_set_of_sample(sample_B)

            add_B_flg = True
            for word in word_set_B:
                if b_words_count[word] - a_words_count[word] >= max_gap:
                    add_B_flg = False

            if add_B_flg:
                b_candidates.append(sample_B)
                for word in word_set_B:
                    b_words_count[word] += 1
                break
    return a_candidates, b_candidates