from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
data = pd.read_csv("movie_info.csv")

"""
=================== Body =============================
"""


class Movie(BaseModel):
    movie_id: int
    movie_title: str
    release_date: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =

# show four genres
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children"]}

# show all generes
'''
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}
'''




@app.post("/api/movies")
def get_movies(genre: list):
    print(genre)
    query_str = " or ".join(map(map_genre, genre))
    results = data.query(query_str)
    results.loc[:, 'score'] = None
    results = results.sample(18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
    print("movies_file")
    print(movies)

    #get the orderbyscore of movies_list
    movies_orderbyscore = sorted(movies, key=lambda i: i.score, reverse=True)
    iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)

    # two dist can store the iid and score
    iid_list = []
    score_iist = []
    # We only choose those movies which have beed scored
    for i in movies_orderbyscore:
        if i.score != 0:
            iid_list.append(i.movie_id)
            score_iist.append(i.score)
    print("iid_list")
    print(iid_list)
    print("score_list")
    print(score_iist)
    #res = get_initial_items(iid,score)
    res = get_initial_items_3_by_onehot_contenbased(iid_list,score_iist)
    # the res is an dataframe of generate recommender of new uesr
    print("res")
    print(res)
    if len(res) > 6:
        res_12 = res[0:6]
    else:
        res_12 = res
    print("res_12")
    print(res_12)

    rec_movies = data.loc[data['movie_id'].isin(res_12['movie_id'])]
    print("rec_movies")
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    """
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    """
    return json.loads(results.to_json(orient="records"))


@app.get("/api/add_recommend/{item_id}")
async def add_recommend(item_id):
    res = get_similar_items(str(item_id), n=5)
    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))

def get_initial_items_3_by_onehot_contenbased(iid_list,score_iist,n=12):

    # prepering for movies_info
    # getting geners_list,item_rep_matrix,item_rep_vector
    geners_list = ["unknown","Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]

    item_rep_vector = data.drop(['release_date','IMDb URL'],axis=1)
    item_rep_vector = item_rep_vector.fillna(0)
    item_rep_matrix = item_rep_vector[geners_list].to_numpy()

    # adding new user in data,and prepering for rating df
    user_add_2(iid_list, score_iist)
    ratings_df = pd.read_csv('new_u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    #Building user profiles (weighted)
    user_id = 944

    user_rating_df = ratings_df[ratings_df['user_id'] == user_id]
    user_preference_df = user_rating_df.sample(frac=1, random_state=1)
    user_preference_df = user_preference_df.reset_index(drop=True)
    user_profile = build_user_profile(user_id, user_preference_df, item_rep_vector, geners_list, weighted=True,
                                      normalized=True)
    # Step 3: Predicting user interest in items
    rec_result = generate_recommendation_results(user_id, user_profile, item_rep_matrix, data)
    print("This is generate_recommendation for new user")
    print(type(rec_result))
    print(rec_result)
    return rec_result



def get_initial_items_2(iid_list,score_iist,n=12):
    # there are some new rating dataset
    res = []
    # We need to add these rating dataset for the new user
    user_add_2(iid_list, score_iist)
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res
def user_add_2(iid_list, score_iist):
    #In original dataset, there are only 6040 user and we ues 6041 as the new user
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data',index=False)
    with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter='\t')
        data_input = []
        # i is the idd, j is the score
        for i,j in zip(iid_list,score_iist):
            s = [user,str(i),int(j),'0']
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)

#original code of user_add and get_initial_items
def user_add(iid, score):
    #In original dataset, there are only 6040 user and we ues 6041 as the new user
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    # There may be wrong
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter='\t')
        data_input = []
        s = [user,str(iid),int(score),'0']
        data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def get_initial_items(iid, score, n=12):
    res = []
    user_add(iid, score)
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res

def get_similar_items(iid, n=12):
    algo = dump.load('./model')[1]
    inner_id = algo.trainset.to_inner_iid(iid)
    print(inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid


# prepering for contenbased
# Building user profile
def build_user_profile(user_id, user_preference_df, item_rep_vector, feature_list, weighted=True, normalized=True):
    ## A: Edit user preference (e.g., rating data)
    user_preference_df = user_preference_df[['movie_id', 'rating']].copy(deep=True).reset_index(drop=True)
    ## B: Calculate item representation matrix to represent user profiles
    user_movie_rating_df = pd.merge(user_preference_df, item_rep_vector)
    user_movie_df = user_movie_rating_df.copy(deep=True)
    user_movie_df = user_movie_df[feature_list]

    ## C: Aggregate item representation matrix
    rating_weight = len(user_preference_df) * [1]
    if weighted:
        rating_weight = user_preference_df.rating / user_preference_df.rating.sum()

    user_profile = user_movie_df.T.dot(rating_weight)

    if normalized:
        user_profile = user_profile / sum(user_profile.values)

    return user_profile

# generate recommendation results
def generate_recommendation_results(user_id, user_profile,item_rep_matrix, movies_data):
    # Comput the cosine similarity
    u_v = user_profile.values
    print("u_v")
    print(u_v)

    u_v_matrix =[u_v]
    print("u_v_matrix")
    print(u_v_matrix)

    recommendation_table =  cosine_similarity(u_v_matrix,item_rep_matrix)

    recommendation_table_df = movies_data[['movie_id', 'movie_title']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)

    return rec_result