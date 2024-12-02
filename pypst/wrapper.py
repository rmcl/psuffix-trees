from typing import List
from pypst.transition_mat import (
    build_transition_matrix,
    build_alphabet_from_dataset
)
from pypst.pst_learn import pst_learn
from pypst.pst_to_pfa import pst_convert_to_pfa

class PST:
    """Create a probabilistic suffix tree (PST) from a dataset."""

    def __init__(
        self,
        L = 2,
        p_min = 0.0073,
        g_min = .01,
        r = 1.6,
        alpha = 17.5,
        alphabet = None
    ):
        self._L = L
        self._p_min = p_min
        self._g_min = g_min
        self._r = r
        self._alpha = alpha
        self._alphabet = alphabet

    @property
    def alphabet(self):
        return list(self._alphabet)

    @property
    def parameters(self):
        return {
            'L': self._L,
            'p_min': self._p_min,
            'g_min': self._g_min,
            'r': self._r,
            'alpha': self._alpha
        }

    def fit(self, dataset : List[List[str]]):
        """Fit the PST model to the dataset."""

        if hasattr(self, '_pst'):  # If already fitted, raise a warning or error
            raise ValueError("The model has already been fitted. Please create a new instance to fit again.")


        if self._alphabet is None:
            self._alphabet = build_alphabet_from_dataset(dataset)

        results = build_transition_matrix(
            dataset,
            self._L,
            alphabet=self._alphabet
        )

        self._pst = pst_learn(
            results['occurrence_mats'],
            alphabet=self._alphabet,
            N=results['N'],
            L=self._L,
            p_min=self._p_min,
            g_min=self._g_min,
            r=self._r,
            alpha=self._alpha)

    @property
    def tree(self):
        """Return the fit PST"""

        if not hasattr(self, '_pst'):
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method first.")

        return self._pst

    @property
    def pfa(self):
        """Convert the PST to a probabilistic finite automaton (PFA)"""
        if not hasattr(self, '_pst'):
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method first.")

        return pst_convert_to_pfa(self._pst)
