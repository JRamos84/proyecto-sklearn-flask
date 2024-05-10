import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from utils import Utils

class Models:
    def __init__(self):
        # Inicializar modelos y parámetros para la búsqueda en cuadrícula
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }

        self.params = {
            'SVR': {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['auto', 'scale'],
                'C': [1, 5, 10]
            },
            'GRADIENT': {
                'loss': ['squared_error', 'huber'],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        }

    # Método para entrenar modelos y realizar búsqueda en cuadrícula
    def grid_training(self, X, y):
        best_score = float('inf')
        best_model = None

        for name, reg in self.reg.items():
            # Crear una instancia del estimador dentro del bucle
            reg_instance = reg
            grid_reg = GridSearchCV(reg_instance, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        # Exportar el mejor modelo
        utils = Utils()
        utils.model_export(best_model, best_score)
