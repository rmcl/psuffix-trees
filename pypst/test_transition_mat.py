import numpy as np
import pytest
from transition_mat import build_transition_matrix

"""
def test_build_transition_matrix_empty_dataset():
    ""Test with empty dataset and order 3""
    dataset = []
    order = 3
    result = build_transition_matrix(dataset, order)
    assert result is None  # Modify this assertion based on the expected behavior
"""

def test_build_transition_matrix_order_1():
    """Test with non-empty dataset and order 1"""
    dataset = [['A', 'B', 'C'], ['B', 'C', 'D'], ['C', 'D', 'E']]
    order = 1
    result = build_transition_matrix(dataset, order)

    assert list(result['p_starting_symbol']) == [1, 1, 1, 0, 0]
    assert result['alphabet'] == ['A','B','C','D','E']
    assert np.array_equal(result['occurrence_mats'][0].to_dense(), np.array([1, 2, 3, 2, 1]))
    assert np.array_equal(result['occurrence_mats'][1].to_dense(), np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]))


# Test case 3: Test with non-empty dataset and order 2
def test_build_transition_matrix_order_2():
    dataset = [['A', 'B', 'C'], ['B', 'C', 'D'], ['C', 'D', 'E']]
    order = 2
    result = build_transition_matrix(dataset, order)

    assert list(result['p_starting_symbol']) == [1, 1, 1, 0, 0]
    assert result['alphabet'] == ['A','B','C','D','E']

    expected_occurrence_mats_0 = np.array([1, 2, 3, 2, 1])
    assert np.array_equal(result['occurrence_mats'][0].to_dense(), expected_occurrence_mats_0)
    assert np.array_equal(result['occurrence_mats'][1].to_dense(), np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]))
    order2_mat = result['occurrence_mats'][2].to_dense()

    assert order2_mat.sum() == 3
    assert order2_mat[0, 1, 2] == 1
    assert order2_mat[1, 2, 3] == 1
    assert order2_mat[2, 3, 4] == 1

def test_build_trans_mat_2():
    dataset = [
        [ch for ch in e]
        for e in ['AAABCABC', 'CABCAB', 'BCABCA', 'CBACBA', 'ABCABC']
    ]
    order = 2

    result = build_transition_matrix(dataset, order)

    assert np.array_equal(result['occurrence_mats'][0].to_dense(), np.array([12, 10, 10]))
    print(result['occurrence_mats'][2])
    assert np.array_equal(result['occurrence_mats'][1].to_dense(), np.array([
        [2, 7, 1],
        [2, 0, 7],
        [6, 2, 0]
    ]))
    assert np.array_equal(result['occurrence_mats'][2].to_dense(), np.array([
        [[1, 1, 0],
         [0, 0, 6],
         [0, 1, 0]],
        [[0, 0, 1],
         [0, 0, 0],
         [5, 0, 0]],
        [[0, 5, 0],
         [2, 0, 0],
         [0, 0, 0]]
    ]))

if __name__ == '__main__':
    pytest.main()
