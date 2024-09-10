# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install plotly
# !pip install wordcloud
# !pip install plotly-express


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # lectura de datos para los registros de Netflix
    df1 = read_file_csv('netflix_titles.csv')
    lista = ['show_id','type','title','director','cast','country','date_added','release_year','rating','duration','listed_in','description']
    data_exporting(df1,lista,'netflix_train.csv')
    # lectura de datosde ratings
    df2 = read_file_csv('IMDb ratings.csv')
    data_exporting(df2, ['weighted_average_vote'],'netflix_ratings_train.csv')
    # lectura de titulos de peliculas
    df3 = read_file_csv('IMDb movies.csv')
    data_exporting(df3,['title','year','genre'],'netflix_movies_train.csv')
 
if __name__ == "__main__":
    main()
