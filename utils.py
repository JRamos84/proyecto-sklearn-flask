
import joblib
import pandas as pd
class Utils:
    # Método para cargar datos desde un archivo CSV
    def load_from_csv(self, path):
        return pd.read_csv(path)

    # Método para seleccionar características y etiquetas del dataset
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    # Método para exportar el mejor modelo entrenado
    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')
