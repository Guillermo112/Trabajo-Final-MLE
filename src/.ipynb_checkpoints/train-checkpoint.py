# Código de Entrenamiento - Modelo de Recomendación de Peliculas
############################################################################
import pandas as pd
# import xgboost as xgb
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    
    print(filename, ' cargado correctamente')
    
    features=['title','director','cast','listed_in','description']
    filledna=df[features]
    filledna=filledna.fillna('')
    for feature in features:
        filledna[feature] = filledna[feature].apply(clean_data)
    filledna['soup'] = filledna.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(filledna['soup'])

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    filledna=filledna.reset_index()
    indices = pd.Series(filledna.index, index=filledna['title'])  
      
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(cosine_sim2, open(package, 'wb'))
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