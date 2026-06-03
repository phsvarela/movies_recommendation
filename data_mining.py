import numpy as np

# Funcão auxiliar para atualizar as médias das intersecções da matriz.
def update_adj_matriz(users, movies, user_groups, movie_groups, ratings, global_mean):
    # Criar matriz de soma e contagem inicialmente zeradas.
    sum_matriz = np.zeros((user_groups, movie_groups))
    count_matriz = np.zeros((user_groups, movie_groups))
    
    # Mapear os IDs da tabela para seus respectivos grupos.
    user_groups_samples = ratings['userId'].map(users).to_numpy(dtype=int)
    movie_groups_samples = ratings['movieId'].map(movies).to_numpy(dtype=int)

    # Colocar as avaliações em um array.
    ratings_samples = ratings['rating'].to_numpy()
    
    # Somar todas as notas dos grupos.
    np.add.at(sum_matriz, (user_groups_samples, movie_groups_samples), ratings_samples)
    # Contar todas as notas dos grupos.
    np.add.at(count_matriz, (user_groups_samples, movie_groups_samples), 1)
    
    # Criar variáveis para o calculo da média do grupo
    group_sum = sum_matriz.sum(axis=1, keepdims=True)
    group_count = count_matriz.sum(axis=1, keepdims=True)

    # Calcular a média do grupo
    group_mean = np.divide(group_sum, group_count,
                            out=np.full_like(group_sum, global_mean),
                            where=group_count != 0)
    
    # Cria matriz com as médias do grupo como plano b
    security_matriz = np.broadcast_to(group_mean, sum_matriz.shape).copy()

    # Calcular a média usando as matrizes de soma e contagem,
    # Se a contagem for 0, mantém a média como a média do grupo.
    new_matriz = np.divide(sum_matriz, count_matriz,
                           out=security_matriz,
                           where=count_matriz != 0)
    
    return new_matriz

# Função para calculo do residuo total da separação em grupos
def get_total_residue(users, movies, adj_matriz, user_ratings_dict, movie_avg_ratings):

    # Defininir residuo total
    total_residue = 0.0

    # Para cada usuário no dicionário original
    for user_id, user_ratings in user_ratings_dict.items():

        # Acessar o grupo do usuário
        user_group = users[user_id]

        # Para cada filme que o usuário avaliou
        for row in user_ratings:

            # Acessar o grupo do filme
            movie_id = int(row[0])
            rating = row[1]
            movie_group = movies[movie_id]

            # Acessar a média do bloco na matriz adjacente
            avg_block = adj_matriz[user_group][movie_group]

            # Calcular a previsão para o filme após os grupos ja estarem todos definidos
            prevision = movie_avg_ratings[movie_id] + avg_block

            # Somar o resíduo quadrático ao residuo total
            total_residue += (rating - prevision) ** 2

    # Retornar o resíduo total dessa tentativa
    return total_residue

# Função principal para organização de grupos
def arranging_groups(user_groups, movie_groups, ratings, user_ratings_dict, movie_ratings_dict, user_original_means, movie_original_means, movie_avg_ratings, global_mean, epochs=50, patience=5):

    # Ordenar os usuários e filmes da média maior pra menor
    ordened_users = sorted(user_original_means.keys(), key=lambda x: user_original_means[x])
    ordened_movies = sorted(movie_original_means.keys(), key=lambda x: movie_original_means[x])

    # Separar os usuários entre os grupos
    users = {}
    for i, user_id in enumerate(ordened_users):
        users[user_id] = min(int((i / len(ordened_users)) * user_groups), user_groups - 1)

    # Separar os filmes entre os grupos
    movies = {}
    for i, movie_id in enumerate(ordened_movies):
        movies[movie_id] = min(int((i / len(ordened_movies)) * movie_groups), movie_groups - 1)

    # Atualizar a matriz adjacente
    adj_matriz = update_adj_matriz(users, movies, user_groups, movie_groups, ratings, global_mean)

    # Armazenar o menor residuo e as épocas sem melhoria
    best_residue = float('inf')
    epochs_without_improv = 0

    for i in range(epochs):
    ################################################ Fase dos usuários ################################################
    # Para cada usuário.
        for user_id in user_ratings_dict:
            
            # Crir um array vazio para armazenar o erro total de cada grupo.
            total_error = []

            # Buscar as avaliações apenas do usuário iterado.
            user_ratings = user_ratings_dict.get(user_id, np.array([]))

            # Para cada grupo de usuário:
            for group_id in range(user_groups):
                
                # Iniciar soma do resíduo quadrático.
                quadratic_residue_sum = 0

                # Para cada avaliação que o usuário deu:
                for row in user_ratings:
                    
                    # Acessando o movieID e a rating do dicionário de usuários
                    movie_id = int(row[0])
                    rating = row[1]

                    # Calcular a previsão.
                    # (Previsão = Média das notas do filme + Média da matriz de adjacências).
                    prevision = movie_avg_ratings[movie_id] + adj_matriz[group_id][movies[movie_id]]
                    
                    # Calcular o resíduo quadrático.
                    # (Resíduo quadrático = (Nota final - Previsão)².
                    quadratic_residue = (rating - prevision) ** 2
                    
                    # Acumular os resíduos na soma.
                    quadratic_residue_sum += quadratic_residue
                    
                # Adicionar a soma dos resíduos no array do erro total por grupo.
                total_error.append(quadratic_residue_sum)            
                    
            # Usuário muda para o grupo no qual apresentou o menor erro total.
            users[user_id] = np.argmin(total_error)
        
        # Atualizar a matriz de adjacências após os usuários mudarem de grupo.
        adj_matriz = update_adj_matriz(users, movies, user_groups, movie_groups, ratings, global_mean)
        
        ################################################ Fase dos filmes ################################################
        # Para cada filme.
        for movie_id in movie_ratings_dict:
            
            # Crir um array vazio para armazenar o erro total de cada grupo.
            total_error = []

            # Buscar as avaliações apenas do filme iterado.
            movie_ratings = movie_ratings_dict.get(movie_id, np.array([]))

            # Para cada grupo de filme:
            for group_id in range(movie_groups):
                
                # Iniciar soma do resíduo quadrático.
                quadratic_residue_sum = 0

                # Para cada avaliação do filme:
                for row in movie_ratings:

                    # Acessando o userID e a rating do dicionário de filmes
                    user_id = int(row[0])
                    rating = row[1]

                    # Calcular a previsão.
                    # (Previsão = Média das notas do filme + Média da matriz de adjacências).
                    prevision = movie_avg_ratings[movie_id] + adj_matriz[users[user_id]][group_id]
                    
                    # Calcular o resíduo quadrático.
                    # (Resíduo quadrático = (Nota final - Previsão)².
                    quadratic_residue = (rating - prevision) ** 2
                    
                    # Acumular os resíduos na soma.
                    quadratic_residue_sum += quadratic_residue
                    
                # Adicionar a soma dos resíduos no array do erro total por grupo.
                total_error.append(quadratic_residue_sum)      
                    
            # Filme muda para o grupo no qual apresentou o menor erro total.
            movies[movie_id] = np.argmin(total_error)

        # Atualizar a matriz de adjacências após os filmes mudarem de grupo.
        adj_matriz = update_adj_matriz(users, movies, user_groups, movie_groups, ratings, global_mean)

        # Verificando o resíduo total da geração
        current_residue = get_total_residue(users, movies, adj_matriz, user_ratings_dict, movie_avg_ratings)

        # Se o resíduo atual for menor do que o menor residuo
        if current_residue < best_residue:

            # O menor resíduo se torna o resíduo atual
            best_residue = current_residue

            # Resetar o número de épocas sem melhora
            epochs_without_improv = 0

        else:

            # Se não for menor aumenta o número de épocas sem melhora
            epochs_without_improv += 1

        # Se a quantidade de épocas sem melhoras bater o número de paciência, o código retorna do jeito que está
        if epochs_without_improv >= patience:

            break
    
    return best_residue

# Função para a separação de dados e criação de uma grid para definir a quantidade ideal de grupos de usuários e filmes
def get_recording_grid(max_groups, ratings, user_ratings_dict, movie_ratings_dict, user_original_means, movie_original_means, movie_avg_ratings, global_mean):

    # Criar a grid
    recording_grid = []

    # Testa todas as opções possíveis até o número máximo de grupos
    for user_groups in range(2, max_groups + 1):
        for movie_groups in range(2, max_groups + 1):

            # Obter o residuo total da tentativa
            final_residue = arranging_groups(user_groups, movie_groups, ratings, user_ratings_dict, movie_ratings_dict, user_original_means, movie_original_means, movie_avg_ratings, global_mean)

            # Armazenar as informações da tentativa na grid
            recording_grid.append({
                'user_groups': user_groups,
                'movie_groups': movie_groups,
                'residue': final_residue
            })

    return recording_grid

# Função para utilizar a grid de treino para retornar o número ideal de grupos de usuários e filmes
def get_ideal_groups_number(recording_grid, min_improv=0.03):

    # Obter o registro com menor resíduo total
    best_record = min(recording_grid, key=lambda x: x['residue'])
    best_error = best_record['residue']

    # Calcular a quantidade de erro é aceitavel baseado no menor registro
    acceptable_error = best_error / (1 - min_improv)

    # Criar uma matriz com as tentativas aceitaveis
    accepted_matrix = [
        item for item in recording_grid
        if item['residue'] <= acceptable_error
    ]

    # Obter o número ideal grupos buscando a opção com menor número de blocos, utilizando como desempate o residuo e a quantidade grupos de usuários
    ideal_groups_number = min(accepted_matrix, key=lambda x: (x['user_groups'] * x['movie_groups'], x['residue'], x['user_groups']))

    # Separar em variáveis os números obtidos
    ideal_user_groups = ideal_groups_number['user_groups']
    ideal_movie_groups = ideal_groups_number['movie_groups']

    return ideal_user_groups, ideal_movie_groups