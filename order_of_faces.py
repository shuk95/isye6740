import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path, connected_components, minimum_spanning_tree

try:
    # PCA class from Question 4 (optional). If it is not available, a plain
    # numpy PCA is used inside OrderOfFaces.pca() instead.
    from pca import PCA  # noqa: F401
except ImportError:
    PCA = None


# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """
        self.images_path = images_path
        mat = scipy.io.loadmat(images_path)
        images = np.asarray(mat['images'], dtype=float)

        # The .mat file stores one image per COLUMN (4096 x 698).
        # Transpose so each ROW is one image in R^4096 -> (m x 4096).
        if images.shape[0] == 4096 and images.shape[1] != 4096:
            images = images.T
        self.data = images
        self.m, self.n = self.data.shape

        # Pairwise Euclidean distances in R^4096 (m x m), computed once.
        self.distances = cdist(self.data, self.data, metric='euclidean')

    def get_image(self, idx: int) -> np.ndarray:
        """Helper: returns image idx as a 64x64 array (MATLAB column-major order)."""
        return self.data[idx].reshape(64, 64, order='F')

    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """
        D = self.distances
        # Keep an edge (weighted by its Euclidean distance) only if the two
        # points are within epsilon of each other; 0 means "no edge".
        A = np.where(D <= epsilon, D, 0.0)
        np.fill_diagonal(A, 0.0)
        return A

    def get_best_epsilon(self) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.
        """
        # Heuristic: epsilon should be as SMALL as possible (so that edges only
        # join truly local neighbours and the graph follows the manifold rather
        # than short-cutting across it) but LARGE enough that the graph is a
        # single connected component (otherwise some geodesic distances are
        # infinite and ISOMAP breaks).
        #
        # The smallest epsilon that connects every node equals the longest edge
        # of the minimum spanning tree of the complete distance graph.
        mst = minimum_spanning_tree(self.distances).toarray()
        epsilon = float(mst.max())

        # Sanity check: the epsilon-graph is connected at this value.
        n_comp, _ = connected_components(self.get_adjacency_matrix(epsilon) > 0,
                                         directed=False)
        assert n_comp == 1
        return epsilon

    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """
        # Step 1: epsilon-neighbourhood graph weighted by Euclidean distance.
        A = self.get_adjacency_matrix(epsilon)

        # Step 2: geodesic (shortest-path) distances along the graph.
        D = shortest_path(A, method='D', directed=False)
        if np.isinf(D).any():
            raise ValueError("Neighbourhood graph is disconnected; increase epsilon.")

        # Step 3: classical MDS on the geodesic distances.
        m = D.shape[0]
        H = np.eye(m) - np.ones((m, m)) / m          # centering matrix
        C = -0.5 * H @ (D ** 2) @ H                  # Gram matrix
        C = (C + C.T) / 2                            # ensure symmetry

        # Step 4: top-2 eigenvectors, scaled by sqrt(eigenvalues).
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1][:2]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        Z = eigvecs * np.sqrt(np.maximum(eigvals, 0))
        return Z

    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """
        X = self.data
        Xc = X - X.mean(axis=0)
        # Principal directions = right singular vectors of the centred data
        # (equivalent to the top eigenvectors of the covariance matrix).
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:num_dim].T                           # (4096 x num_dim)
        return Xc @ W                                # (m x num_dim)


# -----------------------------------------------------------------------------
# Plotting helpers / script for the homework write-up (Q5 parts 1-3)
# -----------------------------------------------------------------------------
def _spread_indices(Z: np.ndarray, k: int = 12, seed: int = 0) -> list:
    """Pick k points spread across a 2-D layout (farthest-point sampling)."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(Z)))]
    dist = np.linalg.norm(Z - Z[idx[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        idx.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(Z - Z[nxt], axis=1))
    return idx


def _scatter_with_faces(ax, Z, faces, highlight, title, xlabel, ylabel, zoom=0.45):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    ax.scatter(Z[:, 0], Z[:, 1], s=8, c='tab:blue', alpha=0.5)
    ax.scatter(Z[highlight, 0], Z[highlight, 1], s=30, c='red', zorder=3)
    for i in highlight:
        im = OffsetImage(faces.get_image(i), cmap='gray', zoom=zoom)
        ab = AnnotationBbox(im, (Z[i, 0], Z[i, 1]), frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor='red', linewidth=1))
        ax.add_artist(ab)
        ax.annotate(str(i), (Z[i, 0], Z[i, 1]), fontsize=7, color='red',
                    xytext=(0, -22), textcoords='offset points', ha='center')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    faces = OrderOfFaces('data/isomap.mat')
    eps = faces.get_best_epsilon()
    A = faces.get_adjacency_matrix(eps)
    deg = (A > 0).sum(axis=1)
    print(f"Best epsilon (smallest value giving a connected graph): {eps:.4f}")
    print(f"Edges: {int((A > 0).sum() / 2)}, avg degree: {deg.mean():.2f}, "
          f"min degree: {deg.min()}, max degree: {deg.max()}")

    # ---------------- Part 1: neighbourhood graph ----------------
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, cmap='viridis')
    fig.colorbar(im, ax=ax, label='Euclidean distance (0 = no edge)')
    ax.set_title(f'Adjacency matrix of the ε-graph (ε = {eps:.2f})')
    ax.set_xlabel('image index')
    ax.set_ylabel('image index')
    fig.tight_layout()
    fig.savefig('q5_1_adjacency_matrix.png', dpi=150)
    plt.close(fig)

    # Graph drawing (force-directed layout, like Gephi) with faces on nodes.
    import networkx as nx
    G = nx.from_numpy_array(A)
    pos = nx.kamada_kawai_layout(G, weight='weight')
    P = np.array([pos[i] for i in range(faces.m)])
    rows, cols = np.nonzero(np.triu(A))
    segs = np.stack([P[rows], P[cols]], axis=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.add_collection(LineCollection(segs, colors='gray', linewidths=0.3, alpha=0.4))
    sel_graph = _spread_indices(P, k=12)
    _scatter_with_faces(ax, P, faces, sel_graph,
                        f'ε-neighbourhood graph (ε = {eps:.2f}) with sample faces',
                        '', '')
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('q5_1_graph_with_faces.png', dpi=150)
    plt.close(fig)

    # ---------------- Part 2: ISOMAP embedding ----------------
    Z_iso = faces.isomap(eps)
    sel_iso = _spread_indices(Z_iso, k=12)
    fig, ax = plt.subplots(figsize=(12, 10))
    _scatter_with_faces(ax, Z_iso, faces, sel_iso,
                        f'ISOMAP 2-D embedding (ε = {eps:.2f})',
                        'component 1', 'component 2')
    fig.tight_layout()
    fig.savefig('q5_2_isomap_embedding.png', dpi=150)
    plt.close(fig)

    # Epsilon tuning illustration: embeddings for several epsilons.
    eps_list = [eps, 12, 15, 20]
    fig, axes = plt.subplots(1, len(eps_list), figsize=(5 * len(eps_list), 4.5))
    for ax, e in zip(axes, eps_list):
        Z = faces.isomap(e)
        ax.scatter(Z[:, 0], Z[:, 1], s=6, c=Z_iso[:, 0], cmap='coolwarm')
        ax.set_title(f'ε = {e:.2f}  (avg degree {((faces.distances <= e).sum(1) - 1).mean():.1f})')
    fig.suptitle('Effect of ε on the ISOMAP embedding (colour = component 1 at best ε)')
    fig.tight_layout()
    fig.savefig('q5_2_epsilon_tuning.png', dpi=130)
    plt.close(fig)

    # ---------------- Part 3: PCA projection ----------------
    Z_pca = faces.pca(2)
    sel_pca = _spread_indices(Z_pca, k=12)
    fig, ax = plt.subplots(figsize=(12, 10))
    _scatter_with_faces(ax, Z_pca, faces, sel_pca,
                        'PCA projection onto the top 2 principal components',
                        'PC 1', 'PC 2')
    fig.tight_layout()
    fig.savefig('q5_3_pca_projection.png', dpi=150)
    plt.close(fig)

    print("Saved: q5_1_adjacency_matrix.png, q5_1_graph_with_faces.png, "
          "q5_2_isomap_embedding.png, q5_2_epsilon_tuning.png, q5_3_pca_projection.png")
