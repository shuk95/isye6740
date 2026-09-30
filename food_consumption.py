from pca import PCA
import numpy as np
import csv

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class FoodConsumptionPCA:
    """
    This class loads and processes a food consumption dataset for performing PCA
    from two perspectives:
    - Country-based PCA: Countries as samples, foods as features.
    - Food-based PCA: Foods as samples, country consumption patterns as features.
    """

    def __init__(self, input_path="data/food-consumption.csv"):
        """
        Initializes the FoodConsumptionPCA object and loads data from a CSV file.

        Parameters:
        ----------
        input_path : str
            Path to the CSV file containing the food consumption data.
        """
        with open(input_path, newline="") as f:
            rows = [r for r in csv.reader(f) if r]

        header = rows[0]
        # First column holds the country name, remaining columns are food items
        self.foods = [h.strip() for h in header[1:]]
        self.countries = [r[0].strip() for r in rows[1:]]
        self.data = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
        # self.data has shape (num_countries, num_foods) = (16, 20)

    def country_pca(self, num_dim: int) -> np.ndarray:
        """
        Performs PCA where each row represents a country and each column represents a food item.

        This will reduce the feature space (foods) to `num_dim` principal components.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to retain.

        Returns:
        -------
        np.ndarray
            A (num_countries, num_dim) array representing countries in the reduced PCA space.
        """
        return PCA().fit_transform(self.data, num_dim)

    def food_pca(self, num_dim: int) -> np.ndarray:
        """
        Performs PCA where each row represents a food item and each column represents a country.

        This will reduce the country-dimension feature space to `num_dim` principal components.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to retain.

        Returns:
        -------
        np.ndarray
            A (num_foods, num_dim) array representing foods in the reduced PCA space.
        """
        return PCA().fit_transform(self.data.T, num_dim)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fc = FoodConsumptionPCA()

    # Q4.1: countries as samples
    Z = fc.country_pca(2)
    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1])
    for name, (x, y) in zip(fc.countries, Z):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Countries in PC space")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("country_pca.png", dpi=150)

    # Q4.2: foods as samples
    F = fc.food_pca(2)
    plt.figure(figsize=(8, 6))
    plt.scatter(F[:, 0], F[:, 1], color="tab:orange")
    for name, (x, y) in zip(fc.foods, F):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Food items in PC space")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("food_pca.png", dpi=150)
    plt.show()
