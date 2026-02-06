import numpy as np

from scipy.stats import entropy
import warnings


def compute_prob(blocks):
    warnings.warn("This function is Deprecated, please use the probability function instead")
    return probability(blocks)


def probability(blocks: np.ndarray) -> np.ndarray:
    """
    compute the probabilities for the unique values in the array given
    :param blocks: numpy array of different computed model results
    :return:
    """
    blocks = np.round(blocks)
    return np.mean([blocks == _ for _ in np.unique(blocks)], axis=1)


def calculate_ie_masked(prob):
    warnings.warn("This function is Deprecated, please use the information_entropy function instead")
    return information_entropy(prob)


def information_entropy(probabilities: np.ndarray, base=2) -> np.ndarray:
    """
    Calculate the Information Entropy of the provided probabilities array

    Eq 2. from doi.org/10.1016/j.tecto.2011.05.001

    but using scipy implementation for speed and simplicity

    note the paper assumes base two everywhere

    :param base: base for entropy, defaults to 2 as per paper
    :param probabilities: probabilities array, first axis is for each possible layers (M, n cells)
    :return: entropy of the probability array, flattens the first dimension
    """
    return entropy(probabilities, base=base)


def fuzziness(probabilities: np.ndarray) -> float:
    """
    Return the fuzziness of the probability array

    Eq 3. from doi.org/10.1016/j.tecto.2011.05.001
    :param probabilities: probabilities array
    :return: float of fuzziness
    """
    p = probabilities
    fuzz = -np.mean(np.nan_to_num(p * np.log(p) + (1 - p) * np.log(1 - p)))
    return fuzz


def total_model_entropy(ie: np.ndarray) -> float:
    """
    Return the Total Model Information Entropy (the mean of the array)
    Eq 4. from doi.org/10.1016/j.tecto.2011.05.001
    :param ie: information entropy array
    :return:
    """
    return float(np.mean(ie))


class OnlineProbability:
    """
    Compute the probabilities and entropy on the go as we compute the models.
    """
    def __init__(self, n_cells: int, unique_lithologies: np.ndarray):
        self.n_cells = n_cells
        self.unique_lithologies = unique_lithologies.astype(int)
        self.n_samples = 0
        self.probability_field = np.zeros((len(unique_lithologies), n_cells))

    def update(self, lith_block: np.ndarray):
        self.n_samples += 1
        lith_block = np.round(lith_block).astype(int)
        
        # Incremental update of the probability field
        # P_n = ((n-1) * P_{n-1} + indicator) / n
        for l_idx, l_val in enumerate(self.unique_lithologies):
            indicator = (lith_block == l_val).astype(float)
            self.probability_field[l_idx] = ((self.n_samples - 1) * self.probability_field[l_idx] + indicator) / self.n_samples

    @property
    def entropy(self):
        return information_entropy(self.probability_field)


def online_probability(n_cells: int, unique_lithologies: np.ndarray):
    """
    Deprecated, use OnlineProbability class instead
    """
    warnings.warn("This function is Deprecated, please use the OnlineProbability class instead")
    return OnlineProbability(n_cells, unique_lithologies)
