import pandas as pd
from utils import Utils
from models import Models

if __name__ == "__main__":
    # Inicializar instancias de Utils y Models
    utils = Utils()
    models = Models()

    # Cargar datos desde el archivo CSV
    data = utils.load_from_csv('./in/felicidad.csv')

    # Separar características y etiquetas del dataset
    X, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])

    # Entrenar modelos y realizar búsqueda en cuadrícula
    models.grid_training(X, y)

