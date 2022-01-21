import json
import time
import logging
from pathlib import Path
from ontology import Ontology
from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import NearestNeighbors




entities = pd.read_csv('../data/new_entities.csv')

ents = entities[["user", "interest type", "interest"]].to_records(index=False).tolist()
ents_triplets = [(user, types, interests) for user, types, interests in ents]

ontology = Ontology('../data/model_junior_fuck.rdf')
ontology.add_entity_list(ents_triplets)
ontology.save('../data/model_junior_last_extended.rdf')

ontology_v2 = Ontology('../data/model_junior_last_extended.rdf')
interests = ontology_v2.get_all_interests()


usrs=[]
intests=[]
for user, interest, interest_type in interests:
    usrs.append(user)
    intests.append(interest)
    #print((user, interest, interest_type))

unique_usrs=list(set(usrs))
unique_intests=list(set(intests))
print("fuck")
df=pd.DataFrame(index=unique_intests, columns=unique_usrs)
df = df.fillna(0)

for user, interest, interest_type in interests:
    df.loc[interest][user]=1

# copy df
df1 = df.copy()

# find the nearest neighbors using NearestNeighbors(n_neighbors=3)
number_neighbors = 3
n_neighbors = 3
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(df.values)
distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

print(df.columns.tolist())

# convert user_name to user_index
user_index = 0

# t: movie_title, m: the row number of t in df
for m, t in list(enumerate(df.index)):

    # find movies without ratings by user_4
    if df.iloc[m, user_index] == 0:
        sim_movies = indices[m].tolist()
        movie_distances = distances[m].tolist()

        # Generally, this is the case: indices[3] = [3 6 7]. The movie itself is in the first place.
        # In this case, we take off 3 from the list. Then, indices[3] == [6 7] to have the nearest NEIGHBORS in the list.
        if m in sim_movies:
            id_movie = sim_movies.index(m)
            sim_movies.remove(m)
            movie_distances.pop(id_movie)

            # However, if the percentage of ratings in the dataset is very low, there are too many 0s in the dataset.
        # Some movies have all 0 ratings and the movies with all 0s are considered the same movies by NearestNeighbors().
        # Then,even the movie itself cannot be included in the indices.
        # For example, indices[3] = [2 4 7] is possible if movie_2, movie_3, movie_4, and movie_7 have all 0s for their ratings.
        # In that case, we take off the farthest movie in the list. Therefore, 7 is taken off from the list, then indices[3] == [2 4].
        else:
            sim_movies = sim_movies[:n_neighbors - 1]
            movie_distances = movie_distances[:n_neighbors - 1]

        # movie_similarty = 1 - movie_distance
        movie_similarity = [1 - x for x in movie_distances]
        movie_similarity_copy = movie_similarity.copy()
        nominator = 0

        # for each similar movie
        for s in range(0, len(movie_similarity)):

            # check if the rating of a similar movie is zero
            if df.iloc[sim_movies[s], user_index] == 0:

                # if the rating is zero, ignore the rating and the similarity in calculating the predicted rating
                if len(movie_similarity_copy) == (number_neighbors - 1):
                    movie_similarity_copy.pop(s)

                else:
                    movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

            # if the rating is not zero, use the rating and similarity in the calculation
            else:
                nominator = nominator + movie_similarity[s] * df.iloc[sim_movies[s], user_index]

        # check if the number of the ratings with non-zero is positive
        if len(movie_similarity_copy) > 0:

            # check if the sum of the ratings of the similar movies is positive.
            if sum(movie_similarity_copy) > 0:
                predicted_r = nominator / sum(movie_similarity_copy)

            # Even if there are some movies for which the ratings are positive, some movies have zero similarity even though they are selected as similar movies.
            # in this case, the predicted rating becomes zero as well
            else:
                predicted_r = 0

        # if all the ratings of the similar movies are zero, then predicted rating should be zero
        else:
            predicted_r = 0

        # place the predicted rating into the copy of the original dataset
        df1.iloc[m, user_index] = predicted_r


def recommend_movies(user, num_recommended_movies):
    print('The list of the Movies {} Has Watched \n'.format(user))

    for m in df[df[user] > 0][user].index.tolist():
        print(m)

    print('\n')

    recommended_movies = []

    for m in df[df[user] == 0].index.tolist():
        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_movies.append((m, predicted_rating))

    sorted_rm = sorted(recommended_movies, key=lambda x: x[1], reverse=True)

    print('The list of the Recommended Movies \n')
    rank = 1
    recs = []
    for recommended_movie in sorted_rm[:num_recommended_movies]:
        recs.append(recommended_movie[0])
        print('{}: {}'.format(rank, recommended_movie[0]))
        rank = rank + 1

    return recs


def movie_recommender(user_index, num_neighbors, num_recommendation):
    number_neighbors = num_neighbors

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

    #user_index = df.columns.tolist().index(user)

    for m, t in list(enumerate(df.index)):
        if df.iloc[m, user_index] == 0:
            sim_movies = indices[m].tolist()
            movie_distances = distances[m].tolist()

            if m in sim_movies:
                id_movie = sim_movies.index(m)
                sim_movies.remove(m)
                movie_distances.pop(id_movie)

            else:
                sim_movies = sim_movies[:n_neighbors - 1]
                movie_distances = movie_distances[:n_neighbors - 1]

            movie_similarity = [1 - x for x in movie_distances]
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0

            for s in range(0, len(movie_similarity)):
                if df.iloc[sim_movies[s], user_index] == 0:
                    if len(movie_similarity_copy) == (number_neighbors - 1):
                        movie_similarity_copy.pop(s)

                    else:
                        movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

                else:
                    nominator = nominator + movie_similarity[s] * df.iloc[sim_movies[s], user_index]

            if len(movie_similarity_copy) > 0:
                if sum(movie_similarity_copy) > 0:
                    predicted_r = nominator / sum(movie_similarity_copy)

                else:
                    predicted_r = 0

            else:
                predicted_r = 0

            df1.iloc[m, user_index] = predicted_r
    return recommend_movies(user, num_recommendation)

user_entity_prefix = 'http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#'
interest_prefix = 'http://dbpedia.org/resource/'
ontology_check = Ontology('../data/model_junior_last_extended.rdf')
cnt = 0
for i in range(1):
    print(str(df.columns.tolist()[i]))
    recos = movie_recommender(i, 3, 4)
    entstr = str(df.columns.tolist()[i])
    if entstr.startswith(user_entity_prefix):
        sp = entstr.find("#") + 1
        interacts = ontology_check.get_user_interactions(entstr[sp:], 'mention')

        usrs = []
        for user, weight in interacts:
            entr = str(user)
            if entr.startswith(user_entity_prefix):
                sp = entr.find("#") + 1
                usrs.append(entr[sp:])

        other_interest = []
        for l in usrs:
            pls = ontology.get_user_interests(l)
            other_interest = other_interest + pls

        if len(list(set(other_interest) & set(recos)))>0:
            cnt = cnt +1

print(cnt)




