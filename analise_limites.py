import pandas as pd
from data_mining import get_ideal_groups_number

def testar_limites():
    print("Carregando o histórico de 82 minutos...\n")
    # Carrega o CSV salvo e transforma de volta no formato de dicionário
    df_resultados = pd.read_csv("resultado_grid_search.csv")
    recording_grid = df_resultados.to_dict('records')

    # Vamos testar o limite de 1% até 6%
    limites_para_testar = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    print("="*40)
    print(" SIMULAÇÃO DE EXIGÊNCIA DO JUIZ")
    print("="*40)
    
    for limite in limites_para_testar:
        u, m = get_ideal_groups_number(recording_grid, min_improv=limite)
        print(f"Exigindo melhoria de {limite*100:.0f}%  ->  Matriz Ideal: {u}x{m}")

if __name__ == "__main__":
    testar_limites()