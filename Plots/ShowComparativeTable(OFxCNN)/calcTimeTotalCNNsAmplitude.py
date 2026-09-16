import os
import numpy as np
import datetime

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)
base_path = os.path.dirname(os.path.dirname(path))

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7
total_folds = 100


def carregar_tempos():
    of_dir = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f"janelamento_{n_janelamento}")
    
    tempo_total_of_por_ocup = {}
    tempo_total_of_global = 0.0

    for oc in ocupacoes:
        of_file = os.path.join(of_dir, f"results_occupation_{oc}.npz")
        data_of = np.load(of_file)
        t_oc = float(data_of['time_mean']) * total_folds
        tempo_total_of_por_ocup[oc] = t_oc
        tempo_total_of_global += t_oc
   
    cnn3_dir = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_3")
    cnn3_sample = np.load(os.path.join(cnn3_dir, f"results_ocupacao_{ocupacoes[0]}.npz"))
    tempo_total_cnn3_global = float(cnn3_sample['time_mean']) * total_folds

    cnn5_dir = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_5")
    cnn5_sample = np.load(os.path.join(cnn5_dir, f"results_ocupacao_{ocupacoes[0]}.npz"))
    tempo_total_cnn5_global = float(cnn5_sample['time_mean']) * total_folds

    cnn8_dir = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_8")
    cnn8_sample = np.load(os.path.join(cnn8_dir, f"results_ocupacao_{ocupacoes[0]}.npz"))
    tempo_total_cnn8_global = float(cnn8_sample['time_mean']) * total_folds


    print("=" * 65)
    print("TEMPO TOTAL DO EXPERIMENTO (Treino + Reconstrução - 100 Folds)")
    print("=" * 65)
    print(f"Filtro Ótimo (Soma de todas as ocupações) : {datetime.timedelta(seconds=round(tempo_total_of_global))}")
    print(f"CNN-3        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_cnn3_global))}")
    print(f"CNN-5        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_cnn5_global))}")
    print(f"CNN-8        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_cnn8_global))}")
    print("=" * 65)
    
    print("\nDETALHAMENTO DO OF POR OCUPAÇÃO:")
    for oc in ocupacoes:
        t_fmt = str(datetime.timedelta(seconds=round(tempo_total_of_por_ocup[oc])))
        print(f"  - Ocupação {oc:>3}%: {t_fmt}")

carregar_tempos()