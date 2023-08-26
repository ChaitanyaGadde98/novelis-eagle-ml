import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SimplifiedDataExploration:

    def __init__(self, data):
        self.data = data.sample(500)  # Taking a smaller sample for visualization
        self.figures_path = Path("data/figures")
        if not self.figures_path.exists():
            self.figures_path.mkdir()

    def visualize_relationship(self):
        # Selecting a subset of features for visualization
        features_to_plot = self.data.columns[:5].tolist()

        for feature in features_to_plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Good/Bad', y=feature, data=self.data)
            plt.title(f'Relationship between {feature} and Good/Bad')
            plt.savefig(self.figures_path / f"{feature}_boxplot_sampled.png")
            plt.close()
