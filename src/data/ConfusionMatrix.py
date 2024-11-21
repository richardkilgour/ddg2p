import numpy as np
import pandas as pd


class ConfusionMatrix:
    def __init__(self, target_languages):
        self.target_languages = target_languages
        self.predicted_languages = target_languages[:]
        self.matrix = np.zeros((len(target_languages), len(target_languages)), dtype=int)
        # Create a mapping from language codes to indices -
        # not used at the mo, but could speed up all the calls to 'index', below
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(target_languages)}

    def __str__(self):
        df = self.to_dataframe()
        return df.to_string()

    def update(self, predicted_language, target_language):
        if predicted_language not in self.predicted_languages:
            self.predicted_languages.append(predicted_language)
            self.matrix.resize((self.matrix.shape[0], self.matrix.shape[1] + 1), refcheck=False)
        target_index = self.target_languages.index(target_language)
        predicted_index = self.predicted_languages.index(predicted_language)
        self.matrix[target_index, predicted_index] += 1

    def merge(self, other):
        # Ensure both confusion matrices have the same target languages
        if self.target_languages != other.target_languages:
            raise ValueError("Target languages must be the same to merge confusion matrices.")
        for predicted_language in other.predicted_languages:
            if predicted_language not in self.predicted_languages:
                self.predicted_languages.append(predicted_language)
                self.matrix.resize((self.matrix.shape[0], self.matrix.shape[1] + 1), refcheck=False)
        for i, target_language in enumerate(self.target_languages):
            for j, predicted_language in enumerate(other.predicted_languages):
                predicted_index = self.predicted_languages.index(predicted_language)
                self.matrix[i, predicted_index] += other.matrix[i, j]

    def to_dataframe(self):
        return pd.DataFrame(self.matrix, index=self.target_languages, columns=self.predicted_languages)

    def true_positive_rate(self):
        true_positives = np.diag(self.matrix)
        total_samples = np.sum(self.matrix)
        return np.sum(true_positives) / total_samples if total_samples > 0 else 0
