import os
import numpy as np
import datetime


root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)
base_path = os.path.dirname(os.path.dirname(path))

ocupacoes = [10,50,80,100]
n_janelamento = 7

def DefinePath_CNN(ocupacao):
    if ocupacao == 10 or ocupacao == 50:
        CNN = 5
        
    elif ocupacao==80 or ocupacao == 100:
        CNN=3
    
    CNN_data_path_amplitude = os.path.join(base_path, f'RedeNeuralConvolucional',f'CNN_{CNN}', f"results_ocupacao_{ocupacao}.npz")
    CNN_data_path_fase = os.path.join(base_path, f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}', f'CNN_{CNN}', f"phase_cnn_occupation_{ocupacao}.npz")
    CNN_estimated_data_path_fase = os.path.join(base_path, f"RedeNeuralConvolucional_Fase", f'CNN_{CNN}',f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz' )
    cnn_type = f'CNN-{CNN}'
    CNN_data_amplitude = np.load(CNN_data_path_amplitude)
    CNN_data_fase = np.load(CNN_data_path_fase)
    CNN_estimated_data_fase = np.load(CNN_estimated_data_path_fase)

    return CNN_data_path_amplitude, CNN_data_amplitude, CNN_data_fase, CNN_estimated_data_fase, cnn_type

def CalcTotalTimeMedio():
    total_folds = 100
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
    cnn8_data_parcial = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_8")
    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"results_occupation_{ocupacao}.npz")
        cnn8_data_path = os.path.join(cnn8_data_parcial, f'results_ocupacao_{ocupacao}.npz')
        OF_data = np.load(OF_data_path)
        cnn8_data = np.load(cnn8_data_path)
        _, CNN_data_amplitude, _, _, cnn_type = DefinePath_CNN(ocupacao)

        tempo_medio_cnn = CNN_data_amplitude['time_mean']
        tempo_medio_of = OF_data['time_mean']
        tempo_medio_cnn8 = cnn8_data['time_mean']

        tempo_total_cnn_segundos = tempo_medio_cnn * total_folds
        tempo_total_of_segundos = tempo_medio_of * total_folds
        tempo_total_cnn8_segundos = tempo_medio_cnn8 * total_folds

        tempo_formatado_cnn = str(datetime.timedelta(seconds=round(tempo_total_cnn_segundos)))
        tempo_formatado_of = str(datetime.timedelta(seconds=round(tempo_total_of_segundos)))
        tempo_formatado_cnn8 = str(datetime.timedelta(seconds=round(tempo_total_cnn8_segundos)))

        print(f"{30*'-'} AMPLITUDE {30*'-'}\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print(rf'| Metodo  | Tempo (H)')
        print(f"| OF | f'{tempo_formatado_of}'")
        print(f"| {cnn_type} | f'{tempo_formatado_cnn}'")
        print(f"| CNN-8 | f'{tempo_formatado_cnn8}'")


CalcTotalTimeMedio()