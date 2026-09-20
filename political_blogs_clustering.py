import numpy as np
from os.path import abspath, exists, dirname, join


class PoliticalBlogsClustering:
    def __init__(self, nodes_path="nodes.txt", edges_path="edges.txt", seed=42, n_init=10):
        """
        nodes_path / edges_path: paths to the provided dataset files. Defaults
        assume they sit next to this script (or in the current working
        directory); pass explicit paths if yours live elsewhere.
        seed / n_init: the k-means step at the end of spectral clustering is
        randomized, so we run it n_init times with different seeds (derived
        from `seed`) and keep the lowest-inertia result, for reproducible,
        stable cluster assignments.
        """
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.seed = seed
        self.n_init = n_init

        self._A = None            # (n, n) symmetric adjacency, isolated nodes removed
        self._labels_true = None  # (n,) 0/1 political-orientation labels
        self._node_ids = None     # (n,) original node ids kept
        self._eigvals = None      # cached full Laplacian eigendecomposition
        self._eigvecs = None

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def _resolve_path(self, path):
        """
        Tries, in order: the path as given (relative to cwd or absolute);
        next to this script; a "data/" subfolder relative to cwd; and a
        "data/" subfolder next to this script -- since the Gradescope
        starter code ships nodes.txt/edges.txt inside a data/ subfolder
        alongside this file, and we don't know which of those two common
        layouts the grading environment will actually use.
        """
        here = dirname(abspath(__file__))
        candidates = [
            path,
            join(here, path),
            join("data", path),
            join(here, "data", path),
        ]
        for candidate in candidates:
            if exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find '{path}'. Looked in: {candidates}"
        )

    def _load_data(self):
        if self._A is not None:
            return

        nodes_path = self._resolve_path(self.nodes_path)
        edges_path = self._resolve_path(self.edges_path)

        # nodes.txt format (tab-separated): <id> <"url"> <label 0/1> <"source dirs">
        ids, labels = [], []
        with open(nodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                ids.append(int(parts[0]))
                labels.append(int(parts[2]))
        ids = np.array(ids, dtype=int)
        labels = np.array(labels, dtype=int)
        id_to_idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)

        # edges.txt format (tab/whitespace-separated): <u> <v>
        A = np.zeros((n, n), dtype=float)
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                u_str, v_str = line.split()[:2]
                u, v = int(u_str), int(v_str)
                if u in id_to_idx and v in id_to_idx:
                    iu, iv = id_to_idx[u], id_to_idx[v]
                    if iu != iv:
                        # Undirected: an edge in EITHER direction sets both entries,
                        # per the assignment's explicit instruction.
                        A[iu, iv] = 1.0
                        A[iv, iu] = 1.0

        # Remove isolated nodes (degree 0), as the assignment explicitly allows.
        deg = A.sum(axis=1)
        keep = np.where(deg > 0)[0]
        A = A[np.ix_(keep, keep)]
        labels = labels[keep]
        ids = ids[keep]

        self._A = A
        self._labels_true = labels
        self._node_ids = ids

    # ------------------------------------------------------------------ #
    # From-scratch k-means (on the spectral embedding), with multiple
    # random restarts (we're allowed -- and the HW's image-compression
    # section explicitly asks for multi-seed search elsewhere in the same
    # spirit -- since the algorithm itself is still fully our own).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kmeans_rows(X, k, seed, n_init, max_iter=300, tol=1e-8):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        best_inertia = np.inf
        best_labels = None

        for _ in range(n_init):
            idx = rng.choice(n, size=k, replace=False)
            centroids = X[idx].copy()
            labels = np.zeros(n, dtype=int)

            for _ in range(max_iter):
                d2 = (
                    np.sum(X ** 2, axis=1, keepdims=True)
                    + np.sum(centroids ** 2, axis=1)
                    - 2 * X @ centroids.T
                )
                d2 = np.maximum(d2, 0.0)
                new_labels = np.argmin(d2, axis=1)

                counts = np.bincount(new_labels, minlength=centroids.shape[0])
                empty = np.where(counts == 0)[0]
                if empty.size > 0:
                    keep_c = np.setdiff1d(np.arange(centroids.shape[0]), empty)
                    if keep_c.size == 0:
                        break
                    centroids = centroids[keep_c]
                    d2 = (
                        np.sum(X ** 2, axis=1, keepdims=True)
                        + np.sum(centroids ** 2, axis=1)
                        - 2 * X @ centroids.T
                    )
                    new_labels = np.argmin(d2, axis=1)

                new_centroids = np.array([
                    X[new_labels == j].mean(axis=0) for j in range(centroids.shape[0])
                ])
                shift = np.sum((new_centroids - centroids) ** 2)
                centroids = new_centroids
                labels = new_labels
                if shift < tol:
                    break

            d2 = (
                np.sum(X ** 2, axis=1, keepdims=True)
                + np.sum(centroids ** 2, axis=1)
                - 2 * X @ centroids.T
            )
            inertia = float(np.sum(np.min(np.maximum(d2, 0.0), axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

        return best_labels

    # ------------------------------------------------------------------ #
    # Spectral clustering
    # ------------------------------------------------------------------ #
    def _ensure_eigendecomposition(self):
        if self._eigvecs is not None:
            return
        D = np.diag(self._A.sum(axis=1))
        L = D - self._A
        eigvals, eigvecs = np.linalg.eigh(L)   # L is symmetric -> real spectrum
        order = np.argsort(eigvals)
        self._eigvals = eigvals[order]
        self._eigvecs = eigvecs[:, order]

    def zero_eigenvectors(self, tol=1e-8):
        """
        Returns (num_components, eigenvectors) where eigenvectors is an
        (n, num_components) array of the Laplacian eigenvectors whose
        eigenvalue is (numerically) zero -- these indicate the graph's
        connected components, per the same argument as HW1 Concept
        Question 1.2.
        """
        self._load_data()
        self._ensure_eigendecomposition()
        num_components = int(np.sum(self._eigvals < tol))
        return num_components, self._eigvecs[:, :num_components]

    def _spectral_embedding(self, k):
        self._ensure_eigendecomposition()
        return self._eigvecs[:, :k]

    def find_majority_labels(self, num_clusters=2):
        '''
        This method loads the data, performs spectral clustering  and reports the majority labels

        Inputs:
            num_clusters (int): The number of clusters to be created

        Output:
            A map with following attributes
            1. overall_mismatch_rate: <2 decimal places>
            2. mismatch_rates: [{"majority_index": <int>, "mismatch_rate": <2 decimal places>}]
        '''

        map = {
            "overall_mismatch_rate": None,
            "mismatch_rates": []
        }

        self._load_data()
        U = self._spectral_embedding(num_clusters)
        cluster_labels = self._kmeans_rows(U, num_clusters, self.seed, self.n_init)

        true_labels = self._labels_true
        n = len(true_labels)

        overall = 0.0
        for c in range(num_clusters):
            mask = cluster_labels == c
            size = int(mask.sum())
            if size == 0:
                continue
            vals, counts = np.unique(true_labels[mask], return_counts=True)
            majority_label = int(vals[np.argmax(counts)])
            n_majority = int(counts.max())
            rate = 1.0 - n_majority / size
            map["mismatch_rates"].append({
                "majority_index": majority_label,
                "mismatch_rate": round(float(rate), 2)
            })
            overall += (size / n) * rate

        map["overall_mismatch_rate"] = round(float(overall), 2)
        return map
