from typing import Tuple
import numpy as np
from collections import deque
import torch

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

        f_vec_indexes, f_vec_values, f_vec_shape = retrieve_f_sigma_sparse(f_mat, cur_sequence_indexes)

        if len(cur_sequence_indexes) > 1:
            f_suf_indexes, f_suf_values, f_suf_shape = retrieve_f_sigma_sparse(f_mat, cur_sequence_indexes[1:])
        else:
            f_suf_indexes, f_suf_values, f_suf_shape = retrieve_f_sigma_sparse(f_mat, [])

        # Add a small epsilon to avoid division by zero
        eps = torch.finfo(torch.float32).eps

        p_sigma_s_values = f_vec_values / (f_vec_values.sum() + eps)
        p_sigma_suf_values = f_suf_values / (f_suf_values.sum() + eps)

        p_sigma_s = torch.sparse_coo_tensor(f_vec_indexes, p_sigma_s_values, f_vec_shape)
        p_sigma_suf = torch.sparse_coo_tensor(f_suf_indexes, p_sigma_suf_values, f_suf_shape)

        def get_value_at_sparse_tensor_index(index, sparse_tensor):
            sparse_tensor_indices = sparse_tensor._indices()
            sparse_tensor_values = sparse_tensor._values()

            for i in range(sparse_tensor_indices.size(1)):
                cur_index = sparse_tensor_indices[:, i]
                if torch.equal(cur_index, index):
                    return sparse_tensor_values[i]

            return 0

        # get non-zero indexes
        total = 0.0

        p_sigma_s_indices = p_sigma_s._indices()
        p_sigma_s_values = p_sigma_s._values()

        # Loop over each non-zero entry
        for i in range(p_sigma_s_indices.size(1)):  # Iterate over columns of the indices tensor
            index = p_sigma_s_indices[:, i]
            p_sigm_s_val = p_sigma_s_values[i]

            # if the probability of the current symbol is less than the minimum transition probability, skip
            # this is equivalent to the psize check
            if p_sigm_s_val < (1 + alpha) * g_min:
                continue
            else:
                print('PSIZE TRUE', index)

            p_sigma_suf_val = get_value_at_sparse_tensor_index(index, p_sigma_suf)

            cur_index_ratio = (p_sigm_s_val + eps) / (p_sigma_suf_val + eps)
            ratio_test = (cur_index_ratio >= r) or (cur_index_ratio <= 1 / r)
            print(index, ratio_test)
            if ratio_test:
                total += 1

        print(cur_sequence_indexes, total)

        if total > 0 and cur_depth < len(tbar):
            tbar[cur_depth]['string'].append(cur_sequence_indexes)
            node, depth = find_parent(cur_sequence_indexes, tbar)
            tbar[cur_depth]['parent'].append((node, depth))
            tbar[cur_depth]['label'].append(cur_sequence)
            tbar[cur_depth]['internal'].append(0)

        if len(cur_sequence_indexes) < L:
            f_vec_prime_indexes, f_vec_prime_values = retrieve_f_prime_sparse(
                f_mat,
                cur_sequence_indexes)

            #print('ALL INDEXES', f_vec_prime_indexes.shape)
            for i in range(f_vec_prime_indexes.size(1)):
                f_vec_prime_index = f_vec_prime_indexes[:, i]
                f_vec_prime_value = f_vec_prime_values[i].item()

                #print('INDEX', f_vec_prime_index)
                #print('VALUE', f_vec_prime_values)


                p_sigmaprime_s = f_vec_prime_value / (N[cur_depth] + eps)
                if p_sigmaprime_s >= p_min:
                    new_sequence = [alphabet[idx] for idx in f_vec_prime_index]
                    sequence_queue_sbar.append(new_sequence)

            #p_sigmaprime_s = f_vec_prime / (N[cur_depth] + np.finfo(float).eps)
            #add_nodes = np.where(p_sigmaprime_s >= p_min)[0]
            #for j in add_nodes:
            #    # Prepend the new symbol to the current sequence
            #    new_sequence = [alphabet[j]] + cur_sequence
            #    sequence_queue_sbar.append(new_sequence)

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

    return f_mat[len(s)][:, tuple(s)].squeeze()

def retrieve_f_prime_sparse(f_mat, s) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a tuple of indices and values for the sparse tensor"""
    if len(s) == 0:
        return f_mat[1]._indices(), f_mat[1]._values()

    indices = f_mat[len(s)]._indices()
    values = f_mat[len(s)]._values()

    filtered_indices = []
    filtered_values = []
    for i in range(indices.size(1)):
        index = indices[:, i]
        if torch.equal(index[0:len(s)], torch.tensor(s)):
            filtered_indices.append(index[1:])
            filtered_values.append(values[i].item())

    # Convert the filtered indices and values to tensors
    filtered_indices = torch.stack(filtered_indices, dim=1)
    filtered_values = torch.tensor(filtered_values)

    print('S', s)
    print('FILTERED INDICES', filtered_indices)
    print('FILTERED VALUES', filtered_values)
    asdjsl

    # Return the filtered sparse indices and values
    return filtered_indices, filtered_values

    """
    # Convert `s` to a tensor for easier comparison
    s_tensor = torch.tensor(s, device=indices.device)

    # Create a mask to filter indices that match the `s` tuple
    mask = torch.any(indices[1:len(s)+1].T.unsqueeze(1) == s_tensor, dim=-1).all(dim=-1)

    # Filter the indices and values based on the mask
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]

    # Return the filtered sparse indices and values
    return filtered_indices, filtered_values
    """

def retrieve_f_sigma(f_mat, s):
    if len(s) == 0:
        return f_mat[0]
    return f_mat[len(s)][tuple(s)]

def retrieve_f_sigma_sparse(f_mat, s) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int]]:
    """Return a tuple of indices, values and shape"""
    if len(s) == 0:
        return f_mat[0]._indices(), f_mat[0]._values(), f_mat[0].shape

    indices = f_mat[len(s)]._indices()
    values = f_mat[len(s)]._values()
    shape = f_mat[len(s)].shape

    filtered_indices = []
    filtered_values = []
    for i in range(indices.size(1)):
        index = indices[:, i]
        if torch.equal(index[0:len(s)], torch.tensor(s)):
            filtered_indices.append(index[1:])
            filtered_values.append(values[i].item())

    # Convert the filtered indices and values to tensors
    filtered_indices = torch.stack(filtered_indices, dim=1)
    filtered_values = torch.tensor(filtered_values)

    # Return the filtered sparse indices and values
    return filtered_indices, filtered_values, shape[0:len(s)]


def find_gsigma(tbar, f_mat, g_min, N, p_smoothing):
    for i in range(len(tbar)):
        for j in range(len(tbar[i].get('string', []))):
            f_vec_indices, f_vec_values, f_vec_shape = retrieve_f_sigma_sparse(f_mat, tbar[i]['string'][j])

            p_sigma_s_indices = f_vec_indices
            p_sigma_s_values = f_vec_values / (f_vec_values.sum() + np.finfo(float).eps)
            if tbar[i]['string'][j]:
                f = retrieve_f_sparse(f_mat, tbar[i]['string'][j])
                p_s = f / N[len(tbar[i]['string'][j])]
            else:
                f, p_s = 0, 1
            sigma_norm = len(p_sigma_s_values)


            #g_sigma_s = p_sigma_s * (1 - sigma_norm * g_min_sparse) + g_min_sparse
            g_sigma_s_values = []
            for value in p_sigma_s_values:
                g_sigma_s_values.append(value * (1 - sigma_norm * g_min) + g_min)

            g_sigma_s = torch.sparse_coo_tensor(
                p_sigma_s_indices,
                torch.tensor(g_sigma_s_values),
                f_vec_shape)

            p_sigma_s = torch.sparse_coo_tensor(
                p_sigma_s_indices,
                p_sigma_s_values,
                f_vec_shape)

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
        return f_mat[0].sum()
    return f_mat[len(s)][tuple(s)].squeeze()


def retrieve_f_sparse(f_mat, s):
    if len(s) == 0:
        return f_mat[0].sum()

    sparse_tensor = f_mat[len(s)]
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Convert `s` to a tensor for easier comparison
    s_tensor = torch.tensor(s, device=indices.device, dtype=indices.dtype)

    # Mask to find indices matching `s`
    # Note: indices[1:len(s)+1] slices along the sparse dimensions being indexed
    mask = torch.all(indices[:len(s)].T == s_tensor, dim=1)

    # Filter the sparse tensor's indices and values based on the mask
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]

    #print('FILTERED INDICES', filtered_indices)
    #print('FILTERED VALUES', filtered_values)
    #print('FILTERED SHAPE', filtered_values.shape)

    return filtered_values
