import json
import numpy as np
from transition_mat import (
    build_transition_matrix,
    build_alphabet_from_dataset,
)
from pst_learn import (
    pst_learn
)


def test_compare_example_pst():

    with open('fixtures/output_symbols.json', 'r') as fp:
        dataset = json.load(fp)

    with open('fixtures/output_tree.json', 'r') as fp:
        tree = json.load(fp)

    generated_alphabet = build_alphabet_from_dataset(dataset)
    expected_alphabet = [a for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcd']
    assert set(generated_alphabet) == set(expected_alphabet), f"Expected alphabet {expected_alphabet}, but got {generated_alphabet}"

    # use the expected alphabet because MATLAB seems to generate somewhat random
    # alphabet order.
    alphabet = expected_alphabet

    L = 2
    p_min = 0.0073
    g_min = .01
    r = 1.6
    alpha = 17.5
    p_smoothing = 0.0

    transition_matrix = build_transition_matrix(
        dataset,
        2,
        alphabet=alphabet)

    assert len(transition_matrix['occurrence_mats']) == 3, "The occurrence matrices length does not match the expected length."
    assert len(transition_matrix['N']) == 3, "The N length does not match the expected length."
    assert len(transition_matrix['alphabet']) == len(alphabet), "The alphabet length does not match the expected length."

    assert np.allclose(
        transition_matrix['occurrence_mats'][0],
        np.array([1959, 1675, 1468, 2992, 1938, 2035, 790, 1494, 889, 2121, 3214, 736, 2230, 614, 579, 270, 1907, 575, 950, 3205, 529, 121, 3, 1705, 801, 1385, 1223, 1249, 619, 1013])
    ), "The zero order values do not match the expected values."

    assert transition_matrix['occurrence_mats'][1][0, 0] == 3
    assert transition_matrix['occurrence_mats'][1][29, 28] == 2

    generated_tree = pst_learn(
        transition_matrix['occurrence_mats'],
        transition_matrix['alphabet'],
        transition_matrix['N'],
        L=L, p_min=p_min, g_min=g_min, r=r, alpha=alpha, p_smoothing=p_smoothing)

    assert len(generated_tree) == len(tree), f"Generated tree len: {len(generated_tree)} does not match expected tree len: {len(tree)}"

    assert generated_tree[2]['string'] == [
        [7, 3],
        [5, 10],
        [9, 10],
        [26, 10],
        [0, 19],
        [23, 19],
        [24, 19],
        [10, 26],
        [6, 27]
    ], 'Comparing the 2nd node strings did not match.'

    assert generated_tree[2]['parent'] == [
        (3, 1), (10, 1), (10, 1), (10, 1), (17, 1), (17, 1), (17, 1), (22, 1), (23, 1)
    ]
