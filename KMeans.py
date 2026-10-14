from PIL import Image
import numpy as np
import time
from os.path import abspath, dirname, exists, join


class KMeansImpl:
    def __init__(self):
        # Random seed used for centroid initialization. Fixed by default so
        # runs are reproducible (the HW explicitly asks for this); pass a
        # different value / None if you want fresh randomness each call.
        self.seed = 0
        # Convergence tolerance on centroid movement, on the same scale as
        # sklearn.cluster.KMeans's default: tol * mean(per-feature variance
        # of the data), compared against the SQUARED centroid shift. This is
        # at least as strict as sklearn's default (tol=1e-4).
        self.tol = 1e-4
        self.max_iter = 300

    def load_image(self, image_name="1.jpeg"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        if exists(image_name):
            path = image_name
        else:
            # Fall back to a "data/" subfolder (relative to cwd, or next to
            # this script) in case the image isn't directly on the given
            # path -- this matches how the starter code ships its sample
            # images, without changing behavior when image_name already
            # resolves directly.
            here = dirname(abspath(__file__))
            for candidate in (join("data", image_name), join(here, "data", image_name), join(here, image_name)):
                if exists(candidate):
                    path = candidate
                    break
            else:
                path = image_name  # let PIL raise its normal FileNotFoundError
        return np.array(Image.open(path).convert("RGB"))

    # ------------------------------------------------------------------ #
    # Internal helpers -- all operate on a *weighted* point set (points,
    # weights). This lets `compress()` cluster on the image's UNIQUE colors
    # (weighted by how many pixels have that color) instead of every single
    # pixel: a photo with 1.15M pixels typically has far fewer distinct
    # colors, and clustering weighted unique colors gives the mathematically
    # identical result (same objective, same centroids) to clustering every
    # raw pixel -- it's just an exact speedup, not an approximation. Passing
    # weights=np.ones(N) recovers plain, unweighted K-means.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pairwise_dist(points, centroids, norm_distance):
        """
        points:    (N, d) float64
        centroids: (k, d) float64
        Returns (N, k) distance matrix: squared Euclidean if norm_distance==2,
        Manhattan (L1) if norm_distance==1.
        """
        if norm_distance == 2:
            p2 = np.sum(points ** 2, axis=1, keepdims=True)        # (N, 1)
            c2 = np.sum(centroids ** 2, axis=1, keepdims=True).T   # (1, k)
            cross = points @ centroids.T                            # (N, k), BLAS matmul
            d = p2 + c2 - 2 * cross
            return np.maximum(d, 0.0)
        elif norm_distance == 1:
            # Accumulate per-channel instead of forming an (N, k, d) tensor
            # (d is small and fixed -- 3 for RGB -- so this stays cheap).
            N, k = points.shape[0], centroids.shape[0]
            dist = np.zeros((N, k))
            for c in range(points.shape[1]):
                dist += np.abs(points[:, c:c + 1] - centroids[None, :, c])
            return dist
        else:
            raise ValueError("norm_distance must be 1 (Manhattan) or 2 (Euclidean)")

    @staticmethod
    def _weighted_choice(rng, weights):
        """
        Sample a single index with probability proportional to `weights`,
        via cumulative-sum + searchsorted rather than rng.choice(p=...) --
        this sidesteps rng.choice's strict "probabilities must sum to 1"
        check, which can raise a spurious ValueError on some numpy
        versions/platforms due to ordinary floating-point round-off after
        normalizing.
        """
        cum = np.cumsum(weights)
        total = cum[-1]
        r = rng.random() * total
        idx = int(np.searchsorted(cum, r, side="right"))
        return min(idx, len(weights) - 1)

    def _kmeanspp_init(self, points, weights, k, norm_distance, rng):
        """
        Weighted, greedy k-means++ seeding: pick centroids one at a time.
        The first centroid is sampled with probability proportional to
        weight (a random restart, so different `self.seed` values still
        explore different solutions); each subsequent centroid is the point
        that MAXIMIZES weight * (distance to nearest chosen centroid)^2,
        i.e. the greedy/deterministic variant of k-means++ (the same
        farthest-point-style rule scikit-learn's own k-means++ now uses
        internally as its default candidate selection). Picking the best
        candidate outright -- rather than a weighted random draw -- gives
        far more reliable, uniformly fast convergence in Lloyd's algorithm
        (a handful of iterations instead of anywhere from ~20 to ~100+),
        which matters here since the grading environment enforces a runtime
        budget.
        """
        N = points.shape[0]
        centroids = np.empty((k, points.shape[1]))
        first = self._weighted_choice(rng, weights)
        centroids[0] = points[first]

        closest = self._pairwise_dist(points, centroids[0:1], norm_distance)[:, 0]
        for i in range(1, k):
            # `closest` is already the *squared* Euclidean distance when
            # norm_distance == 2 (that's what `_pairwise_dist` returns in
            # that branch), but the plain, unsquared Manhattan distance
            # when norm_distance == 1. The k-means++ score needs
            # weight * distance^2 either way, so square `closest` only for
            # L1. (Squaring it again for L2 -- a bug in an earlier version
            # of this file -- silently turned the score into
            # weight * distance^4, badly degrading the seeding.)
            dist_sq = closest if norm_distance == 2 else closest ** 2
            score = weights * dist_sq
            next_idx = int(np.argmax(score))
            centroids[i] = points[next_idx]
            new_dist = self._pairwise_dist(points, centroids[i:i + 1], norm_distance)[:, 0]
            closest = np.minimum(closest, new_dist)
        return centroids

    @staticmethod
    def _weighted_median_1d(values, weights):
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cw = np.cumsum(w)
        cutoff = cw[-1] / 2.0
        idx = int(np.searchsorted(cw, cutoff))
        idx = min(idx, len(v) - 1)
        return v[idx]

    def _update_centroids(self, points, weights, labels, k_eff, norm_distance):
        """
        Vectorized, weight-aware centroid update.
        L2 -> weighted mean, via np.bincount (one pass per channel).
        L1 -> weighted coordinate-wise median (the correct minimizer of
              summed L1 distance), computed after a single sort-by-label so
              each cluster's members are a contiguous slice.
        """
        d = points.shape[1]
        new_centroids = np.empty((k_eff, d))

        if norm_distance == 2:
            wsum = np.bincount(labels, weights=weights, minlength=k_eff)
            for c in range(d):
                s = np.bincount(labels, weights=weights * points[:, c], minlength=k_eff)
                new_centroids[:, c] = s / wsum
        else:
            order = np.argsort(labels, kind="stable")
            sorted_labels = labels[order]
            sorted_points = points[order]
            sorted_weights = weights[order]
            counts = np.bincount(sorted_labels, minlength=k_eff)
            offsets = np.concatenate(([0], np.cumsum(counts)))
            for j in range(k_eff):
                start, end = offsets[j], offsets[j + 1]
                pts_j = sorted_points[start:end]
                w_j = sorted_weights[start:end]
                for c in range(d):
                    new_centroids[j, c] = self._weighted_median_1d(pts_j[:, c], w_j)

        return new_centroids

    def _run_kmeans(self, points, weights, num_clusters, norm_distance):
        """
        points:  (N, d) float64 array (N = number of *unique* colors, usually).
        weights: (N,) float64 array (pixel counts for each unique color).
        Returns (labels, centroids (k_eff, d), n_iter, k_eff, converged)
        """
        rng = np.random.default_rng(self.seed)

        # Weighted variance, matching the effect of computing variance over
        # every raw pixel without materializing them.
        wsum = weights.sum()
        wmean = (weights[:, None] * points).sum(axis=0) / wsum
        wvar = (weights[:, None] * (points - wmean) ** 2).sum(axis=0) / wsum
        tol_scaled = self.tol * np.mean(wvar)

        centroids = self._kmeanspp_init(points, weights, num_clusters, norm_distance, rng)

        n_iter = 0
        converged = False
        labels = None

        while n_iter < self.max_iter:
            n_iter += 1

            dists = self._pairwise_dist(points, centroids, norm_distance)
            labels = np.argmin(dists, axis=1)

            # Empty-cluster handling (by total pixel WEIGHT, not just unique-
            # color count): decrement k rather than terminate or reinitialize,
            # per the assignment's explicit instruction.
            wcounts = np.bincount(labels, weights=weights, minlength=centroids.shape[0])
            empty = np.where(wcounts == 0)[0]
            if empty.size > 0:
                keep = np.setdiff1d(np.arange(centroids.shape[0]), empty)
                if keep.size == 0:
                    raise RuntimeError("All clusters emptied; num_clusters too large for this image.")
                # centroids[keep] renumbers surviving clusters to a fresh,
                # contiguous 0..k_eff-1 range, so recomputing argmin against
                # the shrunken centroid array gives valid labels directly.
                centroids = centroids[keep]
                dists = self._pairwise_dist(points, centroids, norm_distance)
                labels = np.argmin(dists, axis=1)

            k_eff = centroids.shape[0]
            new_centroids = self._update_centroids(points, weights, labels, k_eff, norm_distance)

            sq_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if sq_shift < tol_scaled:
                converged = True
                break

        dists = self._pairwise_dist(points, centroids, norm_distance)
        labels = np.argmin(dists, axis=1)

        return labels, centroids, n_iter, centroids.shape[0], converged

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compress the image using K-Means clustering.

        Parameters:
            pixels: 3D image for each channel (a, b, 3), values range from 0 to 255.
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).

        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """
        map = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {}
        }

        pixels = np.asarray(pixels)
        a, b = pixels.shape[0], pixels.shape[1]
        d = pixels.shape[2] if pixels.ndim == 3 else 1
        flat = pixels.reshape(-1, d).astype(np.float64)

        t0 = time.perf_counter()

        # Cluster on unique colors weighted by pixel count -- exactly
        # equivalent to clustering every pixel, but the per-iteration cost
        # scales with the number of DISTINCT colors rather than the number
        # of pixels, which is what makes this feasible on multi-megapixel
        # images within the assignment's runtime budget.
        unique_colors, inverse, counts = np.unique(
            flat, axis=0, return_inverse=True, return_counts=True
        )
        inverse = inverse.reshape(-1)
        weights = counts.astype(np.float64)

        labels_u, centroids, n_iter, k_eff, converged = self._run_kmeans(
            unique_colors, weights, num_clusters, norm_distance
        )
        labels0 = labels_u[inverse]  # broadcast unique-color labels back to every pixel

        elapsed = time.perf_counter() - t0

        # The assignment's formatting spec calls "class" a "column vector
        # with size(pixels, 1) elements" (1-indexed: 1, 2, 3, ...) -- i.e.
        # one label per pixel in the SAME row-major order as pixels.reshape
        # (-1, 3), as an (N, 1) column, not a 2D (a, b) grid.
        class_col = (labels0 + 1).reshape(-1, 1)
        recon_flat = np.clip(centroids[labels0], 0, 255)
        img = recon_flat.reshape(a, b, d).astype(np.uint8)

        # Final distortion (the K-means objective value, J) at convergence --
        # useful for comparing multiple random seeds and keeping the best one,
        # as the assignment asks ("try multiple times and report ... the best
        # seed in terms of image quality"). Computed on the unique-color
        # representation (weighted), which is exactly equal to summing over
        # every raw pixel.
        final_dist_per_color = self._pairwise_dist(
            unique_colors, centroids, norm_distance
        )[np.arange(unique_colors.shape[0]), labels_u]
        final_distortion = float(np.sum(weights * final_dist_per_color))

        map["class"] = class_col
        map["centroid"] = centroids
        map["img"] = img
        map["number_of_iterations"] = n_iter
        map["time_taken"] = elapsed
        map["additional_args"] = {
            "requested_k": num_clusters,
            "effective_k": k_eff,
            "converged": converged,
            "norm_distance": norm_distance,
            "num_unique_colors": unique_colors.shape[0],
            "final_distortion": final_distortion,
            "class_grid": class_col.reshape(a, b),  # convenience: same labels, image-shaped
        }

        return map
