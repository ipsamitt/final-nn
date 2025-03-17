# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from random import choices

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pos_idxs = [i for i, x in enumerate(labels) if x]
    neg_idxs = [i for i, x in enumerate(labels) if not x]

    pos_seqs = [seqs[i] for i in pos_idxs]
    neg_seqs = [seqs[i] for i in neg_idxs]

    pos_count = len(pos_idxs)
    neg_count = len(neg_idxs)
    diff = abs(pos_count - neg_count)
    additional_labels = []
    additional_samples = []

    if pos_count > neg_count:
        additional_samples = choices(neg_seqs, k=diff)
        additional_labels = [False] * diff

    elif neg_count > pos_count:
        additional_samples = choices(pos_seqs, k=diff)
        additional_labels = [True] * diff


    sampled_seqs = seqs + additional_samples
    sampled_labels = (labels + additional_labels)

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    A = [1, 0, 0, 0]
    T = [0, 1, 0, 0]
    C = [0, 0, 1, 0]
    G = [0, 0, 0, 1]

    encoded_list = []

    for seq in seq_arr:
        encoded_seq = []
        for char in seq:
            if char == "A":
                encoded_seq = encoded_seq + (A)
            if char == "C":
                encoded_seq = encoded_seq +(C)
            if char == "T":
                encoded_seq = encoded_seq +(T)
            if char == "G":
                encoded_seq = encoded_seq +(G)
        encoded_list.append(encoded_seq)
    return encoded_list