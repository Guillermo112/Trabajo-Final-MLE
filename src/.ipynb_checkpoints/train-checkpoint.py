# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################
import pandas as pd
# import xgboost as xgb
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df
    #X_train = df.drop(['DEFAULT'],axis=1)
    #y_train = df[['DEFAULT']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    # print(df.columns())
    #Removiendo stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    X_train['description'] = X_train['description'].fillna('')  
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(X_train['description'])
    #Output the shape of tfidf_matrix
    tfidf_matrix.shape
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(X_train.index, index=X_train['title']).drop_duplicates()
    
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(cosine_sim, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')

def get_recommendations(title, cosine_sim):
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# Entrenamiento completo
def main():
    read_file_csv('netflix_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()