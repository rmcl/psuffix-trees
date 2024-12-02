from typing import List, Dict
import numpy as np
import torch

def convert_sequence_to_indexes(alphabet, sequence):
    """Convert a sequence of characters to their corresponding indexes in the alphabet."""
    return [alphabet.index(char) for char in sequence]

def build_alphabet_from_dataset(dataset : List[List[str]]) -> List[str]:
    """Iterate through the dataset and build an alphabet of unique items."""
    alphabet = set()
    ordered_alphabet = []
    for row in dataset:
        for item in row:
            if item not in alphabet:
                ordered_alphabet.append(item)
                alphabet.add(item)

    """
    # todo - remove this int conversion if we go to a string based alphabet

    def sort_numeric_if_possible(x):
        try:
            return float(x)
        except ValueError:
            return x

    sorted_alphabet = sorted(list(alphabet), key=sort_numeric_if_possible)
    alphabet_map = {
        item: idx
        for idx, item in enumerate(sorted_alphabet)
    }
    """

    return ordered_alphabet


def build_transition_matrix(
    dataset : List[List[str]],
    order : int,
    alphabet : List[str] = None
):
    """Build a set of transition matrices for a given dataset and order.

    Inputs:
        dataset (List[List[str]]) - a list of sequences of items
        order (int) - the order of the PST to build
        alphabet (List[str]) - an optional list of items. The position in the list is the index in the alphabet
            if not provided, the alphabet will be built from the dataset

    Outputs:
        N [alphabet size, 1] - vector of total entries for each order
        p_starting_symbol (Pi) [alphabet size, 1] - probability distribution over the possible starting states (or symbols) in a sequence. It represents the likelihood of each symbol being the first one observed in a sequence.
    """

    if alphabet is None:
        alphabet = build_alphabet_from_dataset(dataset)

    alphabet_length = len(alphabet)

    """
    occurrence_mats = [
        np.zeros((alphabet_length,) * (i+1), dtype=np.uint16)
        for i in range(order + 1)
    ]"""

    # Create a dictionary of all the non-zero counts we find
    # indexed by order, then by the tuple of co-occuring syllables
    occurence_mats_counts = [
        {}
        for i in range(order + 1)
    ]

    p_starting_symbol = np.zeros(alphabet_length, dtype=np.uint16)
    n = np.zeros(order + 1, dtype=np.uint32)

    for cur_sequence in dataset:
        for cur_item_index in range(len(cur_sequence)):

            cur_item = cur_sequence[cur_item_index]
            cur_item_alphabet_index = alphabet.index(cur_item)

            # If this is the first item in the sequence, increment the starting symbol count
            if cur_item_index == 0:
                # Increment the occurrence count of the current syllable
                p_starting_symbol[cur_item_alphabet_index] += 1

            # Iterate through each order of the PST and increment the co-occurrence count
            # of the current syllable with the next order syllables
            # co_occuring_indexes ends up being a list of indexes of the next order syllables
            # that can be used to dereference the current element of the co-occurrence matrix
            co_occuring_indexes = []
            for cur_order in range(0, order + 1):
                # If the sequence is shorter than the current order, skip
                if len(cur_sequence) <= cur_item_index + cur_order:
                    continue

                # increment the total count of sequences of this order
                n[cur_order] += 1

                next_order_item = cur_sequence[cur_item_index + cur_order]
                next_order_syllables_alphabet_index = alphabet.index(next_order_item)

                co_occuring_indexes.append(next_order_syllables_alphabet_index)

                cur_mat_index = tuple(co_occuring_indexes)
                occurence_mats_counts[cur_order][cur_mat_index] = \
                    occurence_mats_counts[cur_order].get(cur_mat_index, 0) + 1


    # Convert the dictionary of counts to a list of sparse tensors
    occurrence_mats = []
    for i in range(order + 1):
        shape = (alphabet_length,) * (i+1)

        indices = []
        values = []

        for key, value in occurence_mats_counts[i].items():
            indices.append(key)
            values.append(value)

            assert len(shape) == len(key), "The shape of the key should match the order of the matrix"

        indices = torch.tensor(indices, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.int32)

        occurrence_mats.append(
            torch.sparse_coo_tensor(indices.t(), values, size=shape)
        )

    return {
        "occurrence_mats": occurrence_mats,
        "p_starting_symbol": p_starting_symbol,
        "alphabet": alphabet,
        "N": n
    }
