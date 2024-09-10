###################################
# Sistema de Recomendacion (predict.py)
###################################

from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Stopwords

def matrix_stopwords(df, feature):
	tfidf = TfidfVectorizer(stop_words='english')
        #Replace NaN with an empty string
	df[feature] = df[feature].fillna('')
	#Construct the required TF-IDF matrix by fitting and transforming the data
	tfidf_matrix = tfidf.fit_transform(df[feature])
	return tfidf_matrix
 
# Compute the cosine similarity matrix
def matrix_similarity(matrix):
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(netflix_overall.index, index=netflix_overall['title']).drop_duplicates()

#Sistema de Recomendacion
def get_recommendations(title, cosine_sim=cosine_sim):
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

def main():
	tfidf_matrix = matrix_stopwords('netflix_train.csv', ['description'])	

if __name__ == "__main__":
    main()