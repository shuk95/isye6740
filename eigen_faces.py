# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run()
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.
# -----------------------------------------------------------------------------

import os
import glob

import numpy as np
from PIL import Image



# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# Gradescope will depend on the structure of this class as defined. 
# -----------------------------------------------------------------------------
class EigenFacesResult:
    """    
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(subject_1_eigen_faces[0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(subject_2_eigen_faces[0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22
        
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    # Yale faces are 320 (wide) x 243 (tall). Downsampling by a factor of 4 gives
    # 80 x 60.75 -> 80 x 60. PIL uses (width, height); numpy uses (rows, cols).
    IMG_SIZE_PIL = (80, 60)      # (width, height) for PIL.Image.resize
    IMG_SHAPE = (60, 80)         # (height, width) numpy shape of each image
    NUM_EIGENFACES = 6
    RESAMPLE = Image.BILINEAR    # standard pixel interpolation (bilinear)

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
        """
        self.images_root_directory = images_root_directory

        # Training data matrices: one ROW per vectorized image -> shape (m, 4800)
        self.train_1, self.train_files_1 = self._load_training_images("subject01")
        self.train_2, self.train_files_2 = self._load_training_images("subject02")

        # Test images, vectorized -> shape (4800,)
        self.test_1 = self._load_image(os.path.join(images_root_directory, "subject01-test.gif"))
        self.test_2 = self._load_image(os.path.join(images_root_directory, "subject02-test.gif"))

    # ------------------------------------------------------------------
    # Helper methods (preprocessing)
    # ------------------------------------------------------------------
    def _load_image(self, path) -> np.ndarray:
        """Open one image, convert to grayscale, downsample by 4 to 60x80, and vectorize."""
        img = Image.open(path).convert("L")                # grayscale (GIFs may open in palette mode)
        img = img.resize(self.IMG_SIZE_PIL, self.RESAMPLE)  # 320x243 -> 80x60 (PIL: width x height)
        arr = np.asarray(img, dtype=np.float64)             # float to avoid uint8 overflow; shape (60, 80)
        return arr.reshape(-1)                              # vectorize -> (4800,)

    def _load_training_images(self, subject_prefix):
        """Load every image of one subject EXCEPT the '-test' image. Returns (m, 4800) matrix."""
        pattern = os.path.join(self.images_root_directory, f"{subject_prefix}*.gif")
        files = sorted(f for f in glob.glob(pattern) if "test" not in os.path.basename(f).lower())
        if not files:
            raise FileNotFoundError(f"No training images found for {subject_prefix} in {self.images_root_directory}")
        X = np.vstack([self._load_image(f) for f in files])  # each row = one vectorized picture
        return X, files

    # ------------------------------------------------------------------
    # Helper methods (PCA + residual)
    # ------------------------------------------------------------------
    def _pca(self, X, k):
        """
        PCA on data matrix X (m x d, one image per row).

        Steps: mean -> center -> covariance -> eigendecomposition -> sort -> keep top k.

        Returns
        -------
        mean_face : (d,)    average training image
        top_vecs  : (d, k)  top-k unit-norm eigenvectors (eigenfaces) as columns
        top_vals  : (k,)    corresponding eigenvalues, largest first
        """
        m = X.shape[0]
        mean_face = X.mean(axis=0)                  # (d,)
        Xc = X - mean_face                          # centered data, (m, d)
        C = (Xc.T @ Xc) / m                         # sample covariance, (d, d) = (4800, 4800)
        eigvals, eigvecs = np.linalg.eigh(C)        # C is symmetric -> eigh; ascending order
        order = np.argsort(eigvals)[::-1][:k]       # indices of the k largest eigenvalues
        top_vals = eigvals[order]
        top_vecs = eigvecs[:, order]                # columns are unit-norm eigenvectors
        return mean_face, top_vecs, top_vals

    @staticmethod
    def _projection_residual(test_vec, mean_face, eigenface):
        """
        s_ij = || c - e e^T c ||_2^2, where c = test_j - mean_i and e = top eigenface of subject i
        (e has unit norm, so e e^T c is the projection of c onto e).
        """
        c = test_vec - mean_face
        residual = c - eigenface * (eigenface @ c)
        return float(residual @ residual)

    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
            See the assignment for the required downsampling, the images to use
            per subject, the number of eigenfaces, and the residual formula.
        """
        k = self.NUM_EIGENFACES

        # --- Step 1: eigenfaces for each subject ---------------------------------
        mean_1, vecs_1, self.eigenvalues_1 = self._pca(self.train_1, k)
        mean_2, vecs_2, self.eigenvalues_2 = self._pca(self.train_2, k)
        self.mean_face_1, self.mean_face_2 = mean_1, mean_2

        # Reshape each eigenvector (column, length 4800) back into a 60x80 image -> (6, 60, 80)
        eigenfaces_1 = vecs_1.T.reshape(k, *self.IMG_SHAPE)
        eigenfaces_2 = vecs_2.T.reshape(k, *self.IMG_SHAPE)

        # --- Step 2: projection residuals using the single TOP eigenface --------
        top_1 = vecs_1[:, 0]     # subject 1's top eigenface, (4800,)
        top_2 = vecs_2[:, 0]     # subject 2's top eigenface, (4800,)

        # s_ij: test image j projected on subject i's eigenface (i = eigenface subject, j = test image)
        projection_residual_s11 = self._projection_residual(self.test_1, mean_1, top_1)
        projection_residual_s12 = self._projection_residual(self.test_2, mean_1, top_1)
        projection_residual_s21 = self._projection_residual(self.test_1, mean_2, top_2)
        projection_residual_s22 = self._projection_residual(self.test_2, mean_2, top_2)

        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=projection_residual_s11,
            s12=projection_residual_s12,
            s21=projection_residual_s21,
            s22=projection_residual_s22
        )

# -----------------------------------------------------------------------------
# Report helper: plot eigenfaces and print residuals (not used by Gradescope)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ef = EigenFaces("data/yalefaces")
    result = ef.run()

    print("Training images used:")
    print("  Subject 1:", [os.path.basename(f) for f in ef.train_files_1])
    print("  Subject 2:", [os.path.basename(f) for f in ef.train_files_2])
    print("Top-6 eigenvalues, subject 1:", np.round(ef.eigenvalues_1, 2))
    print("Top-6 eigenvalues, subject 2:", np.round(ef.eigenvalues_2, 2))

    # Q6.1 - plot first 6 eigenfaces for each subject
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for row, (faces, name) in enumerate([(result.subject_1_eigen_faces, "Subject 1"),
                                         (result.subject_2_eigen_faces, "Subject 2")]):
        for i in range(6):
            axes[row, i].imshow(faces[i], cmap="gray")
            axes[row, i].set_title(f"{name}\nEigenface {i + 1}", fontsize=9)
            axes[row, i].axis("off")
    plt.tight_layout()
    plt.savefig("eigenfaces_top6.png", dpi=150)
    plt.show()

    # Q6.2 - projection residuals
    print("\nProjection residuals s_ij (i = eigenface subject, j = test image):")
    print(f"  s11 = {result.s11:,.2f}")
    print(f"  s12 = {result.s12:,.2f}")
    print(f"  s21 = {result.s21:,.2f}")
    print(f"  s22 = {result.s22:,.2f}")
    print("\nPrediction: test image j belongs to subject argmin_i s_ij")
    print("  Test 1 ->", "Subject 1" if result.s11 < result.s21 else "Subject 2")
    print("  Test 2 ->", "Subject 1" if result.s12 < result.s22 else "Subject 2")
