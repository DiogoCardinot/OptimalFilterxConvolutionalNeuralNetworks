import os
import numpy as np


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
    cnn_type = f'CNN-{CNN}'
    CNN_data_amplitude = np.load(CNN_data_path_amplitude)
    CNN_data_fase = np.load(CNN_data_path_fase)

    return CNN_data_path_amplitude, CNN_data_amplitude, CNN_data_fase, cnn_type

def ImprimeMetricas_Amplitude():
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"results_occupation_{ocupacao}.npz")
        OF_data = np.load(OF_data_path)
        _, CNN_data_amplitude, _, cnn_type = DefinePath_CNN(ocupacao)
        print(f"{30*'-'} AMPLITUDE {30*'-'}\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print("| Metrica       | RMS      | R^2       | MAE      | MedAE    |")
        print("|" + "-"*15 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
        print(f"| OF            | {OF_data['rms']:.6f} | {OF_data['r2']:.6f} | {OF_data['mae']:.6f} | {OF_data['medae']:.6f} |")
        print(f"| {cnn_type}           | {CNN_data_amplitude['rms']:.6f} | {CNN_data_amplitude['r2']:.6f} | {CNN_data_amplitude['mae']:.6f} | {CNN_data_amplitude['medae']:.6f} |")
        print(100*"=")


def ImprimeMetricas_Fase():
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_OF", f'janelamento_{n_janelamento}')
    real_amplitude_data_parcial = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}')
    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"phase_of_occupation_{ocupacao}.npz")
        OF_data = np.load(OF_data_path)
        Real_Amplitude_data_path = os.path.join(real_amplitude_data_parcial, f"phase_real_amplitude_occupation_{ocupacao}.npz")
        Real_Amplitude_data = np.load(Real_Amplitude_data_path)

        _, _, CNN_data_fase, cnn_type = DefinePath_CNN(ocupacao)
        print(f"{30*'-'} FASE {30*'-'}\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print("| Metrica       | RMS      | R^2       | MAE      | MedAE    |")
        print("|" + "-"*15 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
        print(f"| OF            | {OF_data['rms']:.6f} | {OF_data['r2']:.6f} | {OF_data['mae']:.6f} | {OF_data['medae']:.6f} |")
        print(f"| {cnn_type}           | {CNN_data_fase['rms']:.6f} | {CNN_data_fase['r2']:.6f} | {CNN_data_fase['mae']:.6f} | {CNN_data_fase['medae']:.6f} |")
        print(f"| Real Amplitude           | {Real_Amplitude_data['rms']:.6f} | {Real_Amplitude_data['r2']:.6f} | {Real_Amplitude_data['mae']:.6f} | {Real_Amplitude_data['medae']:.6f} |")
        print(100*"=")


def MelhoriasCNN():
    of_data_parcial_amplitude = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
    of_data_parcial_fase = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_OF", f'janelamento_{n_janelamento}')

    sum_amplitude_rms_of = 0
    sum_amplitude_rms_cnn = 0
    sum_amplitude_std_of = 0
    sum_amplitude_std_cnn = 0

    sum_fase_rms_of = 0
    sum_fase_rms_cnn = 0
    sum_fase_std_of = 0
    sum_fase_std_cnn = 0
    for ocupacao in ocupacoes:
        OF_data_path_amplitude = os.path.join(of_data_parcial_amplitude, f"results_occupation_{ocupacao}.npz")
        OF_data_amplitude = np.load(OF_data_path_amplitude)

        OF_data_path_fase = os.path.join(of_data_parcial_fase, f"phase_of_occupation_{ocupacao}.npz")
        OF_data_fase = np.load(OF_data_path_fase)

        _, CNN_data_amplitude, CNN_data_fase, cnn_type = DefinePath_CNN(ocupacao)
        
        # AMPLITUDE
        # std_error : media do desvio padrao do erro de estimacao para os 100 folds
        of_amplitude_rms = OF_data_amplitude['rms']
        of_amplitude_std = OF_data_amplitude['std_error']
        cnn_amplitude_rms = CNN_data_amplitude['rms']
        cnn_amplitude_std = CNN_data_amplitude['std_error']
        # FASE
        # std : desvio padrao do erro de estimacao na reconstrucao da fase (nao utiliza os folds para a fase, apenas para A tau)
        of_fase_rms = OF_data_fase['rms'] 
        of_fase_std = OF_data_fase['std'] 
        cnn_fase_rms = CNN_data_fase['rms']
        cnn_fase_std = CNN_data_fase['std']

        # AMPLITUDE
        sum_amplitude_rms_of+=of_amplitude_rms
        sum_amplitude_rms_cnn+=cnn_amplitude_rms
        sum_amplitude_std_of+=of_amplitude_std
        sum_amplitude_std_cnn+=cnn_amplitude_std
        # FASE
        sum_fase_rms_of+=of_fase_rms
        sum_fase_rms_cnn+=cnn_fase_rms
        sum_fase_std_of+=of_fase_std
        sum_fase_std_cnn+=cnn_fase_std
    
    total_ocupacoes = len(ocupacoes)
    # AMPLITUDE
    mean_amplitude_rms_of= sum_amplitude_rms_of/total_ocupacoes
    mean_amplitude_rms_cnn=sum_amplitude_rms_cnn/total_ocupacoes
    mean_amplitude_std_of=sum_amplitude_std_of/total_ocupacoes
    mean_amplitude_std_cnn=sum_amplitude_std_cnn/total_ocupacoes
    # FASE
    mean_fase_rms_of=sum_fase_rms_of/total_ocupacoes
    mean_fase_rms_cnn=sum_fase_rms_cnn/total_ocupacoes
    mean_fase_std_of= sum_fase_std_of/total_ocupacoes
    mean_fase_std_cnn=sum_fase_std_cnn/total_ocupacoes

    melhoria_amplitude_rms = ((mean_amplitude_rms_of-mean_amplitude_rms_cnn)/mean_amplitude_rms_of)*100
    melhoria_amplitude_std = ((mean_amplitude_std_of-mean_amplitude_std_cnn)/mean_amplitude_std_of)*100
    melhoria_fase_rms = ((mean_fase_rms_of-mean_fase_rms_cnn)/mean_fase_rms_of)*100
    melhoria_fase_std = ((mean_fase_std_of-mean_fase_std_cnn)/mean_fase_std_of)*100

    # print(r"Amplitude RMS: $\frac{\overline{RMS}_{OF,amp} - \overline{RMS}_{CNN,amp}}{\overline{RMS}_{OF,amp}} \cdot 100$")
    # print(r"Amplitude STD: $\frac{\overline{\sigma}_{OF,amp} - \overline{\sigma}_{CNN,amp}}{\overline{\sigma}_{OF,amp}} \cdot 100$")
    # print(r"Fase RMS:      $\frac{\overline{RMS}_{OF,fase} - \overline{RMS}_{CNN,fase}}{\overline{RMS}_{OF,fase}} \cdot 100$")
    # print(r"Fase STD:      $\frac{\overline{\sigma}_{OF,fase} - \overline{\sigma}_{CNN,fase}}{\overline{\sigma}_{OF,fase}} \cdot 100$")
    print("\n\n")
    print(r"Amplitude RMS: ( (RMS-OF-amp - RMS-CNN-amp) / RMS-OF-amp ) * 100")
    print(r"Amplitude STD: ( (STD-OF-amp - STD-CNN-amp) / STD-OF-amp ) * 100")
    print(r"Fase RMS: ( (RMS-OF-fase - RMS-CNN-fase) / RMS-OF-fase ) * 100")
    print(r"Fase STD: ( (STD-OF-fase - STD-CNN-fase) / STD-OF-fase ) * 100")
    print(f"Melhoria CNN vs OF - Amplitude RMS: {melhoria_amplitude_rms:.1f}%")
    print(f"Melhoria CNN vs OF - Amplitude STD: {melhoria_amplitude_std:.1f}%")
    print(f"Melhoria CNN vs OF - Fase RMS:      {melhoria_fase_rms:.1f}%")
    print(f"Melhoria CNN vs OF - Fase STD:      {melhoria_fase_std:.1f}%")


# ImprimeMetricas_Amplitude()
# ImprimeMetricas_Fase()

MelhoriasCNN()