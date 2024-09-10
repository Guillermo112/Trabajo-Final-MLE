###################################
# Sistema de Recomendacion (predict.py)
###################################
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel


#Sistema de Recomendacion
def get_recommendations(title, cosine_sim):

    filledna = pd.read_csv(os.path.join('../data/processed/netflix_train.csv'))
    indices = pd.Series(filledna.index, index=filledna['title'])
    
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
    return filledna['title'].iloc[movie_indices]

def score_model(filename):
    
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre las recomendaciones    

    for a in df['title']:
        res = get_recommendations(a,cosine_sim=model)
        res.to_csv(os.path.join('../data/scores/', a))
        print(res)
    
    print('Las recomendaciones se ha exportado correctamente en la carpeta scores')
    

def main():
    score_model('netflix_test.csv')
	

if __name__ == "__main__":
    main()