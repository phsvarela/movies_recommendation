import pandas as pd
import numpy as np

# Definir rotas
url = "./ml-latest-small"
ratings_path = url + "/ratings.csv"
users_final_path = url + "/users_final.csv"
movies_final_path = url + "/movies_final.csv"

# Dataframe principal
ratings = pd.read_csv(ratings_path)

# Definir a quantidade de grupos de usuários e de filmes.
user_groups = 7
movie_groups = 7

# Criar os dicionários de médias para usuários e filmes.
user_avg_ratings = ratings.groupby('userId')['rating'].mean().to_dict()
movie_avg_ratings = ratings.groupby('movieId')['rating'].mean().to_dict()

# Criar um dicionário vazio para cada grupo.
users = {}
movies = {}

# Distribuir os usuários entre os grupos por módulo.
for user_id in user_avg_ratings:
    users[user_id] = hash(user_id) % user_groups
    
# Distribuir os filmes entre os grupos por módulo.
for movie_id in movie_avg_ratings:
    movies[movie_id] = hash(movie_id) % movie_groups

# Funcão auxiliar para atualizar as médias das intersecções da matriz.
def update_adj_matriz():
    # Criar matriz de soma e contagem inicialmente zeradas.
    sum_matriz = np.zeros((user_groups, movie_groups))
    count_matriz = np.zeros((user_groups, movie_groups))
    
    # Mapear os IDs da tabela para seus respectivos grupos.
    user_groups_samples = np.array([users[uid] for uid in ratings['userId']]).astype(int)
    movie_groups_samples = np.array([movies[mid] for mid in ratings['movieId']]).astype(int)
    
    # Colocar as avaliações em um array.
    ratings_samples = ratings['rating'].to_numpy()
    
    # Somar todas as notas dos grupos.
    np.add.at(sum_matriz, (user_groups_samples, movie_groups_samples), ratings_samples)
    # Contar todas as notas dos grupos.
    np.add.at(count_matriz, (user_groups_samples, movie_groups_samples), 1)
    
    # Calcular a média usando as matrizes de soma e contagem,
    # Se a contagem for 0, mantém a média como 0.
    new_matriz = np.divide(sum_matriz, count_matriz,
                           out=np.zeros_like(sum_matriz),
                           where=count_matriz != 0)
    
    return new_matriz

# Criar matriz de intersecções inicialmente calculada.
adj_matriz = update_adj_matriz()
epoch = 1

for i in range(epoch):
    ################################################ Fase dos usuários ################################################
    # Para cada usuário.
    for user_id in user_avg_ratings:
        
        # Crir um array vazio para armazenar o erro total de cada grupo.
        total_error = []

        # Buscar as avaliações apenas do usuário iterado.
        user_ratings = ratings.query(f'userId == {user_id}')

        # Para cada grupo de usuário:
        for group_id in range(user_groups):
            
            # Iniciar soma do resíduo quadrático.
            quadratic_residue_sum = 0

            # Para cada avaliação que o usuário deu:
            for rating_sample in user_ratings.itertuples():
                # Calcular a previsão.
                # (Previsão = Média das notas do usuário + Média das notas do filme - Média da matriz de adjacências).
                prevision = (user_avg_ratings[user_id] + movie_avg_ratings[rating_sample.movieId] - adj_matriz[group_id][movies[rating_sample.movieId]])
                
                # Calcular o resíduo quadrático.
                # (Resíduo quadrático = (Nota final - Previsão)².
                quadratic_residue = (rating_sample.rating - prevision) ** 2
                
                # Acumular os resíduos na soma.
                quadratic_residue_sum += quadratic_residue
                
            # Adicionar a soma dos resíduos no array do erro total por grupo.
            total_error.append(quadratic_residue_sum)            
                
        # Usuário muda para o grupo no qual apresentou o menor erro total.
        users[user_id] = np.argmin(total_error)
    
    # Atualizar a matriz de adjacências após os usuários mudarem de grupo.
    adj_matriz = update_adj_matriz()
    
    ################################################ Fase dos filmes ################################################
    # Para cada filme.
    for movie_id in movie_avg_ratings:
        
        # Crir um array vazio para armazenar o erro total de cada grupo.
        total_error = []

        # Buscar as avaliações apenas do filme iterado.
        movie_ratings = ratings.query(f'movieId == {movie_id}')

        # Para cada grupo de filme:
        for group_id in range(movie_groups):
            
            # Iniciar soma do resíduo quadrático.
            quadratic_residue_sum = 0

            # Para cada avaliação do filme:
            for rating_sample in movie_ratings.itertuples():
                # Calcular a previsão.
                # (Previsão = Média das notas do usuário + Média das notas do filme - Média da matriz de adjacências).
                prevision = (user_avg_ratings[rating_sample.userId] + movie_avg_ratings[movie_id] - adj_matriz[users[rating_sample.userId]][group_id])
                
                # Calcular o resíduo quadrático.
                # (Resíduo quadrático = (Nota final - Previsão)².
                quadratic_residue = (rating_sample.rating - prevision) ** 2
                
                # Acumular os resíduos na soma.
                quadratic_residue_sum += quadratic_residue
                
            # Adicionar a soma dos resíduos no array do erro total por grupo.
            total_error.append(quadratic_residue_sum)      
                
        # Filme muda para o grupo no qual apresentou o menor erro total.
        movies[movie_id] = np.argmin(total_error)
    
    # Atualizar a matriz de adjacências após os filmes mudarem de grupo.
    adj_matriz = update_adj_matriz()
    
# Exportar para .csv os grupos finais
pd.DataFrame(users.items(), columns=['userId', 'groupId']).to_csv(users_final_path, index=False)
pd.DataFrame(movies.items(), columns=['movieId', 'groupId']).to_csv(movies_final_path, index=False)