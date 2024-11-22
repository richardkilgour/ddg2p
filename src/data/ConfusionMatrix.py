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
            self.matrix = np.c_[self.matrix, np.zeros(self.matrix.shape[0])]
        target_index = self.target_languages.index(target_language)
        predicted_index = self.predicted_languages.index(predicted_language)
        self.matrix[target_index, predicted_index] += 1

    def update_bulk(self, predicted_language, target_language, value):
        for _ in range(value):
            self.update(predicted_language, target_language)

    def merge(self, other):
        # Ensure the target confusion matrix includes the other target and predicted languages
        for target_language in other.target_languages:
            if target_language not in self.target_languages:
                self.target_languages.append(target_language)
                self.matrix = np.r_[self.matrix, [np.zeros(self.matrix.shape[1])]]
        for predicted_language in other.predicted_languages:
            if predicted_language not in self.predicted_languages:
                self.predicted_languages.append(predicted_language)
                self.matrix = np.c_[self.matrix, np.zeros(self.matrix.shape[0])]
        for i, target_language in enumerate(other.target_languages):
            for j, predicted_language in enumerate(other.predicted_languages):
                predicted_index = self.predicted_languages.index(predicted_language)
                target_index = self.target_languages.index(target_language)
                self.matrix[target_index, predicted_index] += other.matrix[i, j]

    def to_dataframe(self):
        return pd.DataFrame(self.matrix, index=self.target_languages, columns=self.predicted_languages)

    def true_positive_rate(self):
        # Does not work because source and target languages are not sorted!
        true_positives = np.diag(self.matrix)
        total_samples = np.sum(self.matrix)
        return np.sum(true_positives) / total_samples if total_samples > 0 else 0

def main():
    en_cm = ConfusionMatrix(["en_UK"])
    # Correct language
    en_cm.update_bulk("en_UK", "en_UK",3)
    es_cm = ConfusionMatrix(["es_CA", "it"])
    es_cm.update_bulk("es_CA", "es_CA",5)
    es_cm.update_bulk("it", "it",7)
    en_cm.merge(es_cm)
    print(en_cm)
    de_cm = ConfusionMatrix(["en_UK", "de"])
    de_cm.update_bulk("de", "de", 13)
    de_cm.update_bulk("en_UK", "de", 2)
    de_cm.update_bulk("en_UK", "en_UK", 14)
    de_cm.update_bulk("ERROR", "en_UK", 23)
    de_cm.update_bulk("ERROR", "de", 11)
    en_cm.merge(de_cm)
    print(en_cm)

    old_confusion_matrix = ConfusionMatrix(["it"])
    old_confusion_matrix.update_bulk("it", "it", 126)
    old_confusion_matrix.update_bulk("en_UK", "it", 9)
    old_confusion_matrix.update_bulk("es_CA", "it", 38)
    old_confusion_matrix.update_bulk("fr", "it", 11)
    old_confusion_matrix.update_bulk("nl", "it", 1)
    old_confusion_matrix.update_bulk("pt_PO", "it", 7)
    new_confusion_matrix = ConfusionMatrix(["fr"])
    new_confusion_matrix.update_bulk("fr", "fr", 63)
    new_confusion_matrix.update_bulk("en_UK", "fr", 1)
    print(old_confusion_matrix)
    print(new_confusion_matrix)
    old_confusion_matrix.merge(new_confusion_matrix)
    print(old_confusion_matrix)



if __name__ == "__main__":
    main()
