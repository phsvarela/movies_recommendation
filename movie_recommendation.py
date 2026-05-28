import sys
import random
import pandas as pd
from typing import Any, Iterable, cast


def get_movies(userId):
    # Definir rotas.
    url = "./ml-latest-small"
    user_groups_path = url + "/users_final.csv"
    ratings_path = url + "/ratings.csv"
    movies_path = url + "/movies.csv"
    
    # Dataframes.
    ratings = pd.read_csv(ratings_path)    
    users = pd.read_csv(user_groups_path)
    movies = pd.read_csv(movies_path)
    
    # Buscar o ID do grupo e os usuários desse grupo.
    groupId = int(cast(Any, users.loc[users['userId'] == userId, 'groupId']).item())
    group_users = users.loc[users['groupId'] == groupId, 'userId'].values
    
    # Buscar os filmes que o usuário já assistiu.
    watched_by_user = set(cast(Iterable[int], ratings.loc[ratings['userId'] == userId, 'movieId']))
    # Buscar os filmes que o grupo já assistiu.
    watched_by_group = set(ratings.loc[ratings['userId'].isin(group_users), 'movieId'])                        
    # Encontrar os filmes que o usuário ainda não assitiu com os filmes que o grupo já assistiu.
    movies_to_recommend = list(watched_by_group.difference(watched_by_user))    
    
    # Pegar pelo menos os 7 primeiros filmes.
    if len(movies_to_recommend) > 7:        
        movies_to_recommend = random.sample(movies_to_recommend, k=7)
    else:        
        movies_to_recommend = random.sample(movies_to_recommend, k=len(movies_to_recommend))    
        
    movies_title = []
        
    # Buscar o nome desses filmes.
    movies_title = movies[movies['movieId'].isin(movies_to_recommend)]    
    return movies_title['title'].tolist()
                

if len(sys.argv) > 1:    
    movies = get_movies(int(sys.argv[1]))
    print(movies)
else:
    print("É necessário inserir o ID do usuário!")