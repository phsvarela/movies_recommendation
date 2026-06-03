import pandas as pd
import time
from data_mining import get_recording_grid, get_ideal_groups_number

# Definir rotas
url = "./ml-latest-small"
ratings_path = url + "/ratings.csv"
grid_results_path = "resultado_grid_search.csv"

# Dataframe principal
ratings = pd.read_csv(ratings_path)

# Salvando as médias originais de usuários e filmes para a organização de grupos inicial
user_original_means = ratings.groupby('userId')['rating'].mean().to_dict()
movie_original_means = ratings.groupby('movieId')['rating'].mean().to_dict()

# Calculando a nota relativa a média de cada usuário
user_means = ratings.groupby('userId')['rating'].transform('mean')
ratings['rating'] = ratings['rating'] - user_means

# Definir um dicionário para acesso rápido dos ratings de usuários e filmes
user_ratings_dict = (ratings.groupby('userId').apply(lambda x: x[['movieId', 'rating']].to_numpy()).to_dict()) 
movie_ratings_dict = (ratings.groupby('movieId').apply(lambda x: x[['userId', 'rating']].to_numpy()).to_dict())

# Definir a média de todas as notas do sistema
global_mean = ratings['rating'].mean()

# Criar os dicionários de médias para usuários e filmes.
movie_avg_ratings = ratings.groupby('movieId')['rating'].mean().to_dict()

#Função para treinamento do sistema, calcular a matriz mais eficiente para trabalhar os dados
def model_training():

    # Definir a máxima quantidade de grupos de usuários e de filmes.
    max_groups = 15

    # Salvar tempo de inicio
    start_time = time.time()

    # Obter a grid de tentativas de execução
    recording_grid = get_recording_grid(
        max_groups, ratings, user_ratings_dict, movie_ratings_dict,
        user_original_means, movie_original_means, movie_avg_ratings, global_mean
    )

    # Salvar e printar o tempo total levado para criar a grid
    elapsed_time = time.time() - start_time
    print(f" Grid Search concluído em {elapsed_time/60:.2f} minutos.")

    # Salvando o registro de tentativas para não ter que rodar o código novamente
    print("\nSalvando backup do histórico...")
    df_resultados = pd.DataFrame(recording_grid)
    df_resultados.to_csv(grid_results_path, index=False)
    print(f"      Salvo como '{grid_results_path}'.")

    # Obter o o número ideal de grupos
    user_groups, movie_groups = get_ideal_groups_number(recording_grid, min_improv=0.01)

    print("\n" + "="*50)
    print(" VEREDITO DA OTIMIZAÇÃO")
    print("="*50)
    print(f" -> Grupos de Usuários recomendados: {user_groups}")
    print(f" -> Grupos de Filmes recomendados:   {movie_groups}")
    print("="*50)

    # Distribuir os usuários entre os grupos por módulo.
    #for user_id in user_avg_ratings:
        #users[user_id] = hash(user_id) % user_groups
        
    # Distribuir os filmes entre os grupos por módulo.
    #for movie_id in movie_avg_ratings:
        #movies[movie_id] = hash(movie_id) % movie_groups

    # Exportar para .csv os grupos finais
    #pd.DataFrame(best_users.items(), columns=['userId', 'groupId']).to_csv(users_final_path, index=False)
    #pd.DataFrame(best_movies.items(), columns=['movieId', 'groupId']).to_csv(movies_final_path, index=False)

if __name__ == "__main__":
    model_training()