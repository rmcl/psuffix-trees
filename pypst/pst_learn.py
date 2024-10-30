import numpy as np
from collections import deque

def pst_learn(
    f_mat,
    alphabet,
    N,
    L=7,
    p_min=0.0073,
    g_min=0.185,
    r=1.6,
    alpha=17.5,
    p_smoothing=0
):
    """
    PST Learn function based on Ron, Singer, and Tishby's 1996 algorithm "The Power of Amnesia".

    Args:
        f_mat (list): List of frequency tables.
        alphabet (str): String of symbols.
        N (list): Total entries per order.

        L (int): Maximum order (default: 7).
        p_min (float): Minimum occurrence probability (default: 0.0073).
        g_min (float): Minimum transition probability (default: 0.185).
        r (float): Minimum divergence (default: 1.8).
        alpha (float): Smoothing parameter (default: 0).
        p_smoothing (float): Smoothing for probability (default: 0).

    Returns:
        list: A tree array representing the probabilistic suffix tree.
    """

    # Initialize sbar: symbols whose probability >= p_min
    sequence_queue_sbar = [
        [value] for alphabet_index, value in enumerate(alphabet)
        if np.single(f_mat[0][alphabet_index] / N[0]) >= p_min
    ]

    # Initialize tree with empty node
    tbar = [
        {
            'string': [],
            'parent': [],
            'label': [],
            'internal': [],
            'g_sigma_s': [],
            'p': [],
            'f': []
        }
        for _ in range(L + 1)
    ]

    tbar[0].update({
        'string': [[]],
        'parent': [(0, 0)],
        'label': ['epsilon'],
        'internal': [0],
    })

    # Learning process
    while sequence_queue_sbar:
        # this is referred to as S_CHAR in the original code
        cur_sequence = sequence_queue_sbar.pop(0)

        # Convert the sequence to a list of alphabet indexes
        # this is referred to as S_INDEX in the original code
        cur_sequence_indexes = [alphabet.index(item) for item in cur_sequence]

        if len(cur_sequence_indexes) == 0:
            continue

        cur_depth = len(cur_sequence_indexes)

        f_vec = retrieve_f_sigma(f_mat, cur_sequence_indexes)

        if len(cur_sequence_indexes) > 1:
            f_suf = retrieve_f_sigma(f_mat, cur_sequence_indexes[1:])
        else:
            f_suf = retrieve_f_sigma(f_mat, [])

        p_sigma_s = f_vec / (np.sum(f_vec) + np.finfo(float).eps)
        p_sigma_suf = f_suf / (np.sum(f_suf) + np.finfo(float).eps)

        ratio = (p_sigma_s + np.finfo(float).eps) / (p_sigma_suf + np.finfo(float).eps)
        psize = p_sigma_s >= (1 + alpha) * g_min

        ratio_test = (ratio >= r) | (ratio <= 1 / r)
        total = np.sum(ratio_test & psize)

        if total > 0:
            if cur_depth < len(tbar):
                tbar[cur_depth]['string'].append(cur_sequence_indexes)
                node, depth = find_parent(cur_sequence_indexes, tbar)
                tbar[cur_depth]['parent'].append((node, depth))
                tbar[cur_depth]['label'].append(cur_sequence)
                tbar[cur_depth]['internal'].append(0)

        if len(cur_sequence_indexes) < L:
            f_vec_prime = retrieve_f_prime(f_mat, cur_sequence_indexes)
            p_sigmaprime_s = f_vec_prime / (N[cur_depth] + np.finfo(float).eps)
            add_nodes = np.where(p_sigmaprime_s >= p_min)[0]

            for j in add_nodes:
                # Prepend the new symbol to the current sequence
                new_sequence = [alphabet[j]] + cur_sequence
                sequence_queue_sbar.append(new_sequence)

    # Post-process the tree
    tbar = fix_path(tbar)
    tbar = find_gsigma(tbar, f_mat, g_min, N, p_smoothing)

    return tbar


def find_parent(sequence, tbar):
    if len(sequence) == 1:
        return 0, 0

    suffix = sequence[1:]
    parent_depth = len(suffix)

    for idx, candidate in enumerate(tbar[parent_depth]['string']):
        if candidate == suffix:
            return idx, parent_depth

    return 0, 0

def fix_path(tbar, max_iterations=1000):
    iteration = 0
    while iteration < max_iterations:
        changes = False
        iteration += 1
        for i in range(2, len(tbar)):
            for j, curr_string in enumerate(tbar[i].get('string', [])):
                node, depth = find_parent(curr_string, tbar)

                try:
                    parent_depth = tbar[i]['parent'][j][1]
                except IndexError:
                    continue

                if depth > parent_depth:
                    tbar[i]['parent'][j] = (node, depth)
                    parent_depth = depth

                if parent_depth < i - 1:
                    suffix = curr_string[1:]
                    node, depth = find_parent(suffix, tbar)
                    tbar[i - 1]['string'].append(suffix)
                    tbar[i - 1]['parent'].append((node, depth))
                    tbar[i - 1]['label'].append(tbar[i]['label'][j][1:])
                    tbar[i - 1]['internal'].append(1)
                    tbar[i]['parent'][j] = (len(tbar[i - 1]['string']) - 1, i - 1)
                    changes = True
        if not changes:
            break
    return tbar


def retrieve_f_prime(f_mat, s):
    if len(s) == 0:
        return f_mat[1]

    return np.squeeze(f_mat[len(s)][:, tuple(s)])

def retrieve_f_sigma(f_mat, s):
    if len(s) == 0:
        return f_mat[0]
    return f_mat[len(s)][tuple(s)]

def find_gsigma(tbar, f_mat, g_min, N, p_smoothing):
    for i in range(len(tbar)):
        for j in range(len(tbar[i].get('string', []))):
            f_vec = retrieve_f_sigma(f_mat, tbar[i]['string'][j])
            p_sigma_s = f_vec / (np.sum(f_vec) + np.finfo(float).eps)
            if tbar[i]['string'][j]:
                f = retrieve_f(f_mat, tbar[i]['string'][j])
                p_s = f / N[len(tbar[i]['string'][j])]
            else:
                f, p_s = 0, 1
            sigma_norm = len(p_sigma_s)
            g_sigma_s = p_sigma_s * (1 - sigma_norm * g_min) + g_min
            tbar[i]['g_sigma_s'].append(g_sigma_s if p_smoothing else p_sigma_s)
            tbar[i]['p'].append(p_s)
            tbar[i]['f'].append(f)
    return tbar

def retrieve_f(f_mat, s):
    """
    Retrieve the frequency for a given sequence s from the frequency matrix f_mat.

    Parameters:
    - f_mat: A list of frequency matrices for different sequence lengths
    - s: A list of indices representing the sequence (symbols) for which we are retrieving the frequency

    Returns:
    - f: The frequency count for the sequence s
    """
    if len(s) == 0:
        return np.sum(f_mat[0])
    return np.squeeze(f_mat[len(s)][tuple(s)])
