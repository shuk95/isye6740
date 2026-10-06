import numpy as np
from PIL import Image
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class PCA:
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction.

    This class provides a method to compute the principal components of a dataset
    and reduce its dimensionality by projecting the data onto the top components
    that explain the most variance.
    """

    def __init__(self):
        """
        Initializes the PCA class.
        Currently, no parameters are set during initialization.
        """
        # Filled in by fit_transform, useful for plotting / analysis afterwards.
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit_transform(self, data: np.ndarray, num_dim: int) -> np.ndarray:
        """
        Perform PCA on the given dataset and return the first `num_dim` principal components.

        Parameters:
        ----------
        data : np.ndarray
            A (m, n) array where each row is a data point with n features.
        num_dim : int
            The number of principal components to return.

        Returns:
        -------
        np.ndarray
            A (m, num_dim) array representing the data projected onto the top `num_dim`
            principal components. Return the projected data itself -- centred data
            times the top eigenvectors -- with no further rescaling (do not normalise
            each component by its norm and do not whiten by sqrt of the eigenvalue).
        """
        X = np.asarray(data, dtype=float)
        m = X.shape[0]

        # 1. Center the data
        self.mean = X.mean(axis=0)
        Xc = X - self.mean

        # 2. Sample covariance matrix C = (1/m) * Xc^T Xc   (n x n)
        C = (Xc.T @ Xc) / m

        # 3. Eigen-decomposition (C is symmetric, so eigh is appropriate)
        eigvals, eigvecs = np.linalg.eigh(C)

        # 4. Sort by descending eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.eigenvalues = eigvals[:num_dim]
        self.eigenvectors = eigvecs[:, :num_dim]   # (n, num_dim), unit-norm columns

        # 5. Project centred data onto the top eigenvectors
        return Xc @ self.eigenvectors
