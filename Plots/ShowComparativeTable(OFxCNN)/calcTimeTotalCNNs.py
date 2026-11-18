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

# carregar_tempos()

def carregar_tempos_fase():

    # TEMPO PARA ESTIMAR AMPLITUDE
    of_dir = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f"janelamento_{n_janelamento}")
    
    tempo_total_of_por_ocup = {}
    tempo_total_of_global = 0.0

    tempo_total_atau_of_por_ocup = {}
    tempo_total_atau_of_global = 0.0

    #Tempo necessario para estimar A*Tau
    a_tau_path = os.path.join(base_path, "FiltroOtimo", "A_tau_OF", f'janelamento_{n_janelamento}')
    for oc in ocupacoes:
        oc_a_tau_file = os.path.join(a_tau_path, f'results_occupation_{oc}.npz')
        data_of_atau = np.load(oc_a_tau_file)
        t_atau_oc = float(data_of_atau['time_mean'])*total_folds
        tempo_total_atau_of_por_ocup[oc] = t_atau_oc
        tempo_total_atau_of_global += t_atau_oc

    # TEMPO TOTAL DE AMPLITUDE POR OF e CNN
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

    # TEMPO TOTAL PARA DIVISÃO A*Tau/A
    div_atau_of_path = os.path.join(base_path, "FiltroOtimo", 'FaseEstimada_OF', f'janelamento_{n_janelamento}')
    div_atau_ar_path = os.path.join(base_path, "FiltroOtimo", 'FaseEstimada_RealAmplitude', f'janelamento_{n_janelamento}')
    div_atau_cnn3_path = os.path.join(base_path, "FiltroOtimo", 'FaseEstimada_CNN', f'janelamento_{n_janelamento}', 'CNN_3')
    div_atau_cnn5_path = os.path.join(base_path, "FiltroOtimo", 'FaseEstimada_CNN', f'janelamento_{n_janelamento}', 'CNN_5')
    div_atau_cnn8_path = os.path.join(base_path, "FiltroOtimo", 'FaseEstimada_CNN', f'janelamento_{n_janelamento}', 'CNN_8')

    total_time_div_atau_of = 0.0
    total_time_div_atau_ar = 0.0
    total_time_div_atau_cnn3 = 0.0
    total_time_div_atau_cnn5 = 0.0
    total_time_div_atau_cnn8 = 0.0

    for oc in ocupacoes:
        div_atau_of_data = os.path.join(div_atau_of_path, f'phase_of_occupation_{oc}.npz')
        div_atau_of_time = np.load(div_atau_of_data)
        total_time_div_atau_of += div_atau_of_time['div_time_total']

        div_atau_ar_data = os.path.join(div_atau_ar_path, f'phase_real_amplitude_occupation_{oc}.npz')
        div_atau_ar_time = np.load(div_atau_ar_data)
        total_time_div_atau_ar += div_atau_ar_time['div_time_total']

        div_atau_cnn3_data = os.path.join(div_atau_cnn3_path, f'phase_cnn_occupation_{oc}.npz')
        div_atau_cnn3_time = np.load(div_atau_cnn3_data)
        total_time_div_atau_cnn3 += div_atau_cnn3_time['div_time_total']

        div_atau_cnn5_data = os.path.join(div_atau_cnn5_path, f'phase_cnn_occupation_{oc}.npz')
        div_atau_cnn5_time = np.load(div_atau_cnn5_data)
        total_time_div_atau_cnn5 += div_atau_cnn5_time['div_time_total']

        div_atau_cnn8_data = os.path.join(div_atau_cnn8_path, f'phase_cnn_occupation_{oc}.npz')
        div_atau_cnn8_time = np.load(div_atau_cnn8_data)
        total_time_div_atau_cnn8 += div_atau_cnn8_time['div_time_total']

    tempo_total_amplitude_of = tempo_total_atau_of_global + tempo_total_of_global + total_time_div_atau_of
    tempo_total_amplitude_cnn3 = tempo_total_atau_of_global + tempo_total_cnn3_global + total_time_div_atau_cnn3
    tempo_total_amplitude_cnn5 = tempo_total_atau_of_global + tempo_total_cnn5_global + total_time_div_atau_cnn5
    tempo_total_amplitude_cnn8 = tempo_total_atau_of_global + tempo_total_cnn8_global + total_time_div_atau_cnn8
    tempo_total_amplitude_amplitudereal = tempo_total_atau_of_global + total_time_div_atau_ar

    # TEMPO DE ESTIMACAO DIRETA PELAS CNNS
    cnn3_fase_dir = os.path.join(base_path, "RedeNeuralConvolucional_Fase", "CNN_3", f'janelamento_{n_janelamento}')
    cnn3_fase_sample = np.load(os.path.join(cnn3_fase_dir, f"results_ocupacao_{ocupacoes[0]}.npz"))
    tempo_total_fase_cnn3_global = float(cnn3_fase_sample['time_mean']) * total_folds

    cnn5_fase_dir = os.path.join(base_path, "RedeNeuralConvolucional_Fase", "CNN_5", f'janelamento_{n_janelamento}')
    cnn5_fase_sample = np.load(os.path.join(cnn5_fase_dir, f"results_ocupacao_{ocupacoes[0]}.npz"))
    tempo_total_fase_cnn5_global = float(cnn5_fase_sample['time_mean']) * total_folds

    cnn8_fase_dir = os.path.join(base_path, "RedeNeuralConvolucional_Fase", "CNN_8", f'janelamento_{n_janelamento}')
    cnn8_fase_sample = np.load(os.path.join(cnn8_fase_dir, f"results_ocupacao_{ocupacoes[0]}.npz"))
    tempo_total_fase_cnn8_global = float(cnn8_fase_sample['time_mean']) * total_folds

    print("=" * 65)
    print("TEMPO TOTAL DO EXPERIMENTO (Treino + Reconstrução - 100 Folds)")
    print("-" * 65)
    print("--------------------------- ABORDAGEM DE DIVISAO POR AMPLITUDE ---------------------------")
    print(f"Filtro Ótimo (Soma de todas as ocupações) : {datetime.timedelta(seconds=round(tempo_total_amplitude_of))}")
    print(f"Amplitude Real                            : {datetime.timedelta(seconds=round(tempo_total_amplitude_amplitudereal))}")
    print(f"CNN-3        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_amplitude_cnn3))}")
    print(f"CNN-5        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_amplitude_cnn5))}")
    print(f"CNN-8        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_amplitude_cnn8))}")
    print("-" * 65)
    print("--------------------------- ABORDAGEM DE ESTIMACAO DIRETA POR CNN ---------------------------")
    print(f"CNN-3        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_fase_cnn3_global))}")
    print(f"CNN-5        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_fase_cnn5_global))}")
    print(f"CNN-8        (Execução unificada)         : {datetime.timedelta(seconds=round(tempo_total_fase_cnn8_global))}")
    print("=" * 65)

carregar_tempos_fase()
    
    