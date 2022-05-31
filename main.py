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
from surprise import KNNWithMeans
from surprise import SVD
import timeit
from scipy.stats import ttest_rel
import random

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

# getting the gerers_list
columns_data = data.columns.values
geners_list = columns_data[7:26]
print("geners_list")
print(geners_list)

#
list_have_been_recommend_movies = []
# get a empty series to store
User_profile_list = pd.Series()

# paried t test for Contenbased
first_contentbase_rating_result = []
second_contentbase_rating_result = []
first_cf_rating_result = []
second_cf_rating_result = []

SVD_param={}
KNN_param={}
# def gridsearch_param_SVD():
#     reader = Reader()
#     data_df = pd.read_csv('ratings_u.data',sep='\t',names=['uid','iid','rating'])
#     whole_data = Dataset.load_from_df(data_df, reader=reader)
#     SVD_param_grid = {
#         'n_factors': [5,10,20,40,80],
#         'n_epochs': [5,10,15],
#         'biased': [True],
#         }
#     SVD_start=timeit.default_timer()
#     gs_cv = 3
#     SVD_gs = GridSearchCV(SVD, SVD_param_grid, measures=['rmse'], cv=gs_cv)
#     SVD_gs.fit(whole_data)
#     SVD_based_algo = SVD_gs.best_estimator['rmse']
#
#     SVD_best_result = cross_validate(SVD_based_algo, whole_data, measures=['RMSE'], verbose=True)
#
#     SVD_mean_rmse = SVD_best_result['test_rmse'].mean()
#     SVD_runtime=timeit.default_timer()-SVD_start
#     print(SVD_gs.best_params,SVD_mean_rmse,SVD_runtime)
#     return SVD_gs.best_params['rmse']['k'],SVD_gs.best_params['rmse']['sim_options']['name'],SVD_gs.best_params['rmse']['sim_options']['user_based']
#
#
# def gridsearch_param_KNN():
#     reader = Reader()
#     data_df = pd.read_csv('ratings_u.data',sep='\t',names=['uid','iid','rating'])
#     whole_data = Dataset.load_from_df(data_df, reader=reader)
#     KNN_param_grid = {'k': [5,15,35,75,115],
#                     'sim_options': {
#                     'name': ['cosine', 'pearson'],
#                         'user_based': [False,True]
#                     }
#                     }
#
#     KNN_start=timeit.default_timer()
#     KNN_gs = GridSearchCV(KNNWithMeans, KNN_param_grid, measures=['rmse'], cv=gs_cv)
#     KNN_gs.fit(whole_data)
#
#     KNN_algo = KNN_gs.best_estimator['rmse']
#
#     KNN_best_result = cross_validate(KNN_algo, whole_data, measures=['rmse'], verbose=True)
#
#     KNN_mean_rmse = KNN_best_result['test_rmse'].mean()
#     KNN_runtime=timeit.default_timer()-KNN_start
#     print(KNN_gs.best_params, KNN_mean_rmse,KNN_runtime)
#     return KNN_gs.best_params['rmse']['k'],KNN_gs.best_params['rmse']['sim_options']['name'],KNN_gs.best_params['rmse']['sim_options']['user_based']

# SVD_param['n_factors'],SVD_param['n_epochs']=gridsearch_param_SVD()
# KNN_param['k'],KNN_param['name'],KNN_param['user_based']=gridsearch_param_KNN()
SVD_param['n_factors']=40
SVD_param['n_epochs']=5
KNN_param['k']=115
KNN_param['name']='pearson'
KNN_param['user_based']=False


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
    list_genres = list(geners_list)
    random_sample_genre = random.sample(list_genres, 4)
    print(random_sample_genre)
    return {'genre': random_sample_genre}


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
    results = results.sample(
        18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))

@app.post("/api/recommend1")
def get_recommend(movies: List[Movie]):
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
            # adding the rating movies into the have been recommend movies
            list_have_been_recommend_movies.append(i.movie_id)
    #res = get_initial_items(iid,score)
    res = get_initial_items_3_by_onehot_contenbased(iid_list,score_iist)
    # the res is an dataframe of generate recommender of new uesr
    if len(res) > 6:
        res_12 = res[0:6]
    else:
        res_12 = res
    rec_movies = data.loc[data['movie_id'].isin(res_12['movie_id'])]

    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like','IMDb_URL','genres']]

    # We need to add the have been recommended movies
    for i in results['movie_id']:
        list_have_been_recommend_movies.append(i)
    print("The first get_recommend movies result")
    print(results)


    return json.loads(results.to_json(orient="records"))


@app.post("/api/add_recommend1/")
async def add_recommend(movies: List[Movie]):
    print("This is the add_recommend ,let look the input")
    print(movies)

    # get the orderbyscore of movies_list
    movies_orderbyscore = sorted(movies, key=lambda i: i.score, reverse=True)


    # re- recommend
    # The like item should be 5 rating
    print("Now this is the add_recommend function")
    iid_list = []
    score_iist = []

    # We only choose those movies which have beed scored
    for i in movies_orderbyscore:
        if i.score != 0:
            iid_list.append(i.movie_id)
            score_iist.append(i.score)

    score_iist.append(5)

    # We use the new one
    res = get_initial_items_4_by_onehot_contenbased(iid_list, score_iist)
    # the res is an dataframe of generate recommender of new uesr
    if len(res) > 6:
        res_12 = res[0:6]
    else:
        res_12 = res
    rec_movies = data.loc[data['movie_id'].isin(res_12['movie_id'])]
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like','IMDb_URL','genres']]

    # We need to add the have been recommended movies
    for i in results['movie_id']:
        list_have_been_recommend_movies.append(i)

    print("The second get_recommend movies result")
    print(results)

    return json.loads(results.to_json(orient="records"))


# To get the user profile
@app.get("/api/user_profile")
def get_profile_list():
    print("THE iS the API ? USER-PROFILE!!!!!!!!!!!!!!")
    User_profile_list_sorted = User_profile_list.sort_values(ascending=False)
    user_profile_dict = {'genres': User_profile_list_sorted.index, 'value': User_profile_list_sorted.values}
    user_profile_df = pd.DataFrame(user_profile_dict)
    user_profile_df = user_profile_df.iloc[:3]
    json_data = user_profile_df.to_json(orient="records")
    print("json_data")
    print(json_data)
    return json.loads(user_profile_df.to_json(orient="records"))

@app.post("/api/user_profile")
def get_profile_list():
    User_profile_list_sorted = User_profile_list.sort_values(ascending=False)
    user_profile_dict = {'genres': User_profile_list_sorted.index, 'value': User_profile_list_sorted.values}
    user_profile_df = pd.DataFrame(user_profile_dict)
    print("/api/user_profile")
    print(user_profile_df.to_json(orient="records"))
    return json.loads(user_profile_df.to_json(orient="records"))

# This 4 function design for T-test
@app.post("/api/date_for_test1/")
def get_data(movies: List[Movie]):
    print("This is the first time got data for t-test")
    print("This is the first time got data for t-test")
    print("This is the first time got data for t-test")
    a=0
    t=0
    for i in movies:
        a=a+float(i.score)
        t=t+1
    averrating=a/t
    print(averrating)
    
    averrating=str(averrating)
    file = open('ypw_average1.txt', mode='a')
    file.write(averrating)
    file.write('\n')
    file.close()
    return averrating


@app.post("/api/date_for_test2/")
def get_data(movies: List[Movie]):
    print("This is the second time got data for t-test")
    print("This is the second time got data for t-test")
    print("This is the second time got data for t-test")
    a=0
    t=0
    for i in movies:
        a=a+float(i.score)
        t=t+1
    averrating=a/t
    
    averrating=str(averrating)
    print(averrating)
    file = open('ypw_average2.txt', mode='a')
    file.write(averrating)
    file.write('\n')
    file.close()
    return averrating

@app.post("/api/date_for_test1_xjx/")
def get_data(movies: List[Movie]):
    print("This is the first time got data for t-test")
    print("This is the first time got data for t-test")
    print("This is the first time got data for t-test")

    a=0
    t=0
    for i in movies:
        a=a+float(i.score)
        t=t+1
    averrating=a/t
    print(averrating)
    
    averrating=str(averrating)
    file = open('xjx_average1.txt', mode='a')
    file.write(averrating)
    file.write('\n')
    file.close()
    return averrating


@app.post("/api/date_for_test2_xjx/")
def get_data(movies: List[Movie]):
    print("This is the second time got data for t-test")
    print("This is the second time got data for t-test")
    print("This is the second time got data for t-test")
    a=0
    t=0
    for i in movies:
        a=a+float(i.score)
        t=t+1
    averrating=a/t
    averrating=str(averrating)
    print(averrating)
    file = open('xjx_average2.txt', mode='a')
    file.write(averrating)
    file.write('\n')
    file.close()
    return averrating

# use for the first recommend
def get_initial_items_3_by_onehot_contenbased(iid_list,score_iist,n=12):

    # prepering for movies_info
    # getting geners_list,item_rep_matrix,item_rep_vector
    item_rep_vector = data.drop(['release_date','IMDb_URL'],axis=1)
    item_rep_vector = item_rep_vector.fillna(0)
    item_rep_matrix = item_rep_vector[geners_list].to_numpy()

    # adding new user in data,and prepering for rating df
    user_add_2(iid_list, score_iist)
    ratings_df = pd.read_csv('new_yangpeiwen_ratings_u.data', sep='\t', names=['user_id', 'movie_id', 'rating'])
    print("get_initial_items_3_by_onehot_contenbased  ratings_df  ")
    print(ratings_df)

    #Building user profiles (weighted)
    user_id = 6041

    user_rating_df = ratings_df[ratings_df['user_id'] == user_id]
    user_preference_df = user_rating_df.sample(frac=1, random_state=1)
    user_preference_df = user_preference_df.reset_index(drop=True)
    print("get_initial_items_3_by_onehot_contenbased  check the user_preference_df")
    print(user_preference_df)

    user_profile = build_user_profile(user_id, user_preference_df, item_rep_vector, geners_list, weighted=True,
                                      normalized=True)
    print("get_initial_items_3_by_onehot_contenbased  check the User_profile")
    print(user_profile)

    # Global variable User_profile_list update
    global User_profile_list
    User_profile_list = user_profile

    User_profile_list_sorted = User_profile_list.sort_values(ascending=False)
    print("get_initial_items_3_by_onehot_contenbased  User_profile_list_sorted")
    print(User_profile_list_sorted)
    user_profile_dict = {'genres':User_profile_list_sorted.index,'value':User_profile_list_sorted.values}
    print("user_profile_dataframe")
    print(user_profile_dict)
    print("JSON")
    user_profile_df = pd.DataFrame(user_profile_dict)
    print(user_profile_df.to_json(orient="records"))

    print("get_initial_items_3_by_onehot_contenbased  check the item_rep_matrix")
    print(item_rep_matrix)

    # Step 3: Predicting user interest in items
    rec_result = generate_recommendation_results(user_id, user_profile, item_rep_matrix, data)
    print("get_initial_items_3_by_onehot_contenbased : This is generate_recommendation for new user")
    print(type(rec_result))
    print(rec_result)

    # Delete the have been movies
    # rec_result = rec_result.drop(index = rec_result[rec_result['movie_id'].isin(list_have_been_recommend_movies)])
    rec_result = rec_result[~rec_result['movie_id'].isin(list_have_been_recommend_movies)]
    print("get_initial_items_3_by_onehot_contenbased : delete the have been recommended movies")
    print(rec_result)


    return rec_result

#This function is for the first recommend
def user_add_2(iid_list, score_iist):
    """
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
    """

    # In original dataset, there are only 6040 user and we ues 6041 as the new user
    user = 6041
    # simulate adding a new user into the original data file
    df = pd.read_csv('ratings_u.data')
    df.to_csv('new_yangpeiwen_' + 'ratings_u.data', index=False)
    with open(r'new_yangpeiwen_ratings_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        # i is the idd, j is the score
        for i, j in zip(iid_list, score_iist):
            # We dont need the time information of rating
            s = [user, str(i), int(j)]
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)


#use for the first recommend
def get_initial_items_4_by_onehot_contenbased(iid_list,score_iist,n=12):

    # prepering for movies_info
    # getting geners_list,item_rep_matrix,item_rep_vector

    item_rep_vector = data.drop(['release_date','IMDb_URL'],axis=1)
    item_rep_vector = item_rep_vector.fillna(0)
    item_rep_matrix = item_rep_vector[geners_list].to_numpy()

    # adding new user in data,and prepering for rating df
    user_add_3(iid_list, score_iist)
    ratings_df = pd.read_csv('new_yangpeiwen_ratings_u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    #Building user profiles (weighted)
    user_id = 6041

    user_rating_df = ratings_df[ratings_df['user_id'] == user_id]
    user_preference_df = user_rating_df.sample(frac=1, random_state=1)
    user_preference_df = user_preference_df.reset_index(drop=True)
    user_profile = build_user_profile(user_id, user_preference_df, item_rep_vector, geners_list, weighted=True,
                                      normalized=True)

    #Global variable User_profile_list update
    global User_profile_list
    User_profile_list = user_profile

    # Step 3: Predicting user interest in items
    rec_result = generate_recommendation_results(user_id, user_profile, item_rep_matrix, data)

    print("get_initial_items_4_by_onehot_contenbased : This is generate_recommendation for new user")
    print(type(rec_result))
    print(rec_result)

    # Delete the have been movies
    #rec_result = rec_result.drop(index = rec_result[rec_result['movie_id'].isin(list_have_been_recommend_movies)])
    rec_result = rec_result[~rec_result['movie_id'].isin(list_have_been_recommend_movies)]
    print("get_initial_items_4_by_onehot_contenbased : delete the have been recommended movies")
    print(rec_result)

    return rec_result

#This function is for the second recommend
def user_add_3(iid_list, score_iist):
    user = 6041
    with open(r'new_yangpeiwen_ratings_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        # i is the idd, j is the score
        for i, j in zip(iid_list, score_iist):
            # We dont need the time information of rating
            s = [user, str(i), int(j)]
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)


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




# -----------------------------------------------------------------------------------------------------------------

@app.post("/api/recommend2")
def get_recommend2(movies: List[Movie]):
    # print(movies)
    movies_orderbyscore = sorted(movies, key=lambda i: i.score, reverse=True)
    raw_iid_list = []
    score_list = []
    for i in movies_orderbyscore:
        if i.score != 0:
            raw_iid_list.append(i.movie_id)
            score_list.append(i.score)
    # raw_iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    # score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    # res = get_initial_items(raw_iid,score)
    res = get_initial_items2(raw_iid_list, score_list)
    res = [int(i) for i in res]
    if len(res) > 6:
        res = res[0:6]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like','IMDb_URL','genres']]
    print(results)
    for i in results['movie_id']:
        list_have_been_recommend_movies.append(i)

    return json.loads(results.to_json(orient="records"))

@app.post("/api/add_recommend2/")
async def add_recommend2(movies: List[Movie]):
    movies_orderbyscore = sorted(movies, key=lambda i: i.score, reverse=True)
    raw_iid_list = []
    score_list = []
    for i in movies_orderbyscore:
        if i.score != 0:
            raw_iid_list.append(i.movie_id)
            score_list.append(i.score)
    res = get_similar_items2(raw_iid_list, score_list, n=6)
    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like','IMDb_URL','genres']]

    for i in results['movie_id']:
        list_have_been_recommend_movies.append(i)

    return json.loads(results.to_json(orient="records"))


# def user_add(iid, score):
#     user = '6041'
#     # simulate adding a new user into the original data file
#     df = pd.read_csv('./u.data')
#     df.to_csv('new_' + 'u.data',index=False)
#     with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
#         wf = csv.writer(cfa,delimiter='\t')
#         data_input = []
#         s = [user,str(iid),int(score),'0']
#         data_input.append(s)
#         for k in data_input:
#             wf.writerow(k)

def user_add1(raw_iid_list, score_list):
    #In original dataset, there are only 6040 user and we ues 6041 as the new user
    user = '6041'
    # simulate adding a new user into the original data file
    df = pd.read_csv('ratings_u.data')
    df.to_csv('new_xjx_' + 'ratings_u.data',index=False)
    with open(r'new_xjx_ratings_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter='\t')
        data_input = []
        # i is the idd, j is the score
        for i,j in zip(raw_iid_list,score_list):
            s = [user,str(i),int(j),'0']
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def get_initial_items2(raw_iid_list, score_list, n=12):
    start=timeit.default_timer()
    res = []
    user_add1(raw_iid_list, score_list)
    file_path = os.path.expanduser('new_xjx_ratings_u.data')
    reader = Reader(line_format='user item rating', sep='\t')
    # reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    # algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo = SVD(n_factors=SVD_param['n_factors'],n_epochs=SVD_param['n_epochs'],verbose=True)
    algo.fit(trainset)
    # dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in range(3706):
        uid = str(6041)
        raw_iid = str(i)
        pred = algo.predict(uid,raw_iid).est
        all_results[raw_iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    runtime=timeit.default_timer()-start
    print('runtime:',runtime)
    return res

def user_add2(raw_iid_list, score_list):
    user = '6041'

    with open(r'new_xjx_ratings_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        # i is the idd, j is the score
        for i, j in zip(raw_iid_list, score_list):
            s = [user, str(i), int(j), '0']
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def get_similar_items2(raw_iid_list, score_list, n):
    res=[]
    user_add2(raw_iid_list, score_list)
    file_path = os.path.expanduser('new_xjx_ratings_u.data')
    # reader = Reader(line_format='user item rating timestamp', sep='\t')
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    # algo = dump.load('./model')[1]
    algo = KNNWithMeans(k=KNN_param['k'], min_k=1, sim_options={'name': KNN_param['name'], 'user_based': KNN_param['user_based']}, verbose=True)
    algo.fit(trainset)
    all_results = {}
    for i in range(3706):
        if i in list_have_been_recommend_movies:
            continue
        uid = str(6041)
        raw_iid = str(i)
        pred = algo.predict(uid, raw_iid).est
        all_results[raw_iid] = pred
    sorted_list = sorted(all_results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    print(res)
    return res
