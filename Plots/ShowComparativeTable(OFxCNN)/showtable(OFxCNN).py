import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)
base_path = os.path.dirname(os.path.dirname(path))

ocupacoes = [10,50,80,100]
n_janelamento = 7

def DefinePath_CNN(ocupacao, CNN=None):
    if CNN==None and (ocupacao == 10 or ocupacao == 50):
        CNN = 5
        
    elif CNN==None and (ocupacao==80 or ocupacao == 100):
        CNN=3
    
    CNN_data_path_amplitude = os.path.join(base_path, f'RedeNeuralConvolucional',f'CNN_{CNN}', f"results_ocupacao_{ocupacao}.npz")
    CNN_data_path_fase = os.path.join(base_path, f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}', f'CNN_{CNN}', f"phase_cnn_occupation_{ocupacao}.npz")
    CNN_estimated_data_path_fase = os.path.join(base_path, f"RedeNeuralConvolucional_Fase", f'CNN_{CNN}',f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz' )
    cnn_type = f'CNN-{CNN}'
    CNN_data_amplitude = np.load(CNN_data_path_amplitude)
    CNN_data_fase = np.load(CNN_data_path_fase)
    CNN_estimated_data_fase = np.load(CNN_estimated_data_path_fase)

    return CNN_data_path_amplitude, CNN_data_amplitude, CNN_data_fase, CNN_estimated_data_fase, cnn_type

def ImprimeMetricas_Amplitude():
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
    cnn8_data_parcial = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_8")
    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"results_occupation_{ocupacao}.npz")
        cnn8_data_path = os.path.join(cnn8_data_parcial, f'results_ocupacao_{ocupacao}.npz')
        OF_data = np.load(OF_data_path)
        cnn8_data = np.load(cnn8_data_path)
        _, CNN_data_amplitude, _, _, cnn_type = DefinePath_CNN(ocupacao)
        print(f"{30*'-'} AMPLITUDE {30*'-'}\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print("| Metrica       | RMS      | R^2       | MAE      | MedAE    |")
        print("|" + "-"*15 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
        print(f"| OF            | {OF_data['rms']:.6f} | {OF_data['r2']:.6f} | {OF_data['mae']:.6f} | {OF_data['medae']:.6f} |")
        print(f"| {cnn_type}           | {CNN_data_amplitude['rms']:.6f} | {CNN_data_amplitude['r2']:.6f} | {CNN_data_amplitude['mae']:.6f} | {CNN_data_amplitude['medae']:.6f} |")
        print(f"| CNN-8           | {cnn8_data['rms']:.6f} | {cnn8_data['r2']:.6f} | {cnn8_data['mae']:.6f} | {cnn8_data['medae']:.6f} |")
        print(100*"=")


def ImprimeMetricas_Fase():
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_OF", f'janelamento_{n_janelamento}')
    real_amplitude_data_parcial = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}')
    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"phase_of_occupation_{ocupacao}.npz")
        OF_data = np.load(OF_data_path)
        Real_Amplitude_data_path = os.path.join(real_amplitude_data_parcial, f"phase_real_amplitude_occupation_{ocupacao}.npz")
        Real_Amplitude_data = np.load(Real_Amplitude_data_path)

        _, _, CNN_data_fase, CNN_estimated_data_fase, cnn_type = DefinePath_CNN(ocupacao)
        print(f"{30*'-'} FASE {30*'-'}\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print("| Metrica       | RMS      | R^2       | MAE      | MedAE    |")
        print("|" + "-"*15 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
        print(f"| OF            | {OF_data['rms']:.6f} | {OF_data['r2']:.6f} | {OF_data['mae']:.6f} | {OF_data['medae']:.6f} |")
        print(rf'| {cnn_type} (A estimada) | {CNN_data_fase["rms"]:.6f} | {CNN_data_fase["r2"]:.6f} | {CNN_data_fase["mae"]:.6f} | {CNN_data_fase["medae"]:.6f} |')
        print(f"| Real Amplitude           | {Real_Amplitude_data['rms']:.6f} | {Real_Amplitude_data['r2']:.6f} | {Real_Amplitude_data['mae']:.6f} | {Real_Amplitude_data['medae']:.6f} |")
        print(rf'| {cnn_type}        | {CNN_estimated_data_fase['rms']:.6f} | {CNN_estimated_data_fase['r2']:.6f} | {CNN_estimated_data_fase['mae']:.6f} | {CNN_estimated_data_fase['medae']:.6f} |')
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
    sum_fase_rms_cnn_estimated = 0
    sum_fase_std_cnn_estimated = 0
    for ocupacao in ocupacoes:
        OF_data_path_amplitude = os.path.join(of_data_parcial_amplitude, f"results_occupation_{ocupacao}.npz")
        OF_data_amplitude = np.load(OF_data_path_amplitude)

        OF_data_path_fase = os.path.join(of_data_parcial_fase, f"phase_of_occupation_{ocupacao}.npz")
        OF_data_fase = np.load(OF_data_path_fase)

        _, CNN_data_amplitude, CNN_data_fase, CNN_estimated_data_fase, cnn_type = DefinePath_CNN(ocupacao)
        
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
        cnn_estimated_fase_rms = CNN_estimated_data_fase['rms']
        cnn_estimated_fase_std = CNN_estimated_data_fase['std_error']

        # AMPLITUDE
        sum_amplitude_rms_of+=of_amplitude_rms
        sum_amplitude_rms_cnn+=cnn_amplitude_rms
        sum_amplitude_std_of+=of_amplitude_std
        sum_amplitude_std_cnn+=cnn_amplitude_std
        # FASE
        sum_fase_rms_of+=of_fase_rms
        sum_fase_rms_cnn+=cnn_fase_rms
        sum_fase_rms_cnn_estimated+=cnn_estimated_fase_rms

        sum_fase_std_of+=of_fase_std
        sum_fase_std_cnn+=cnn_fase_std
        sum_fase_std_cnn_estimated+=cnn_estimated_fase_std

    
    total_ocupacoes = len(ocupacoes)
    # AMPLITUDE
    mean_amplitude_rms_of= sum_amplitude_rms_of/total_ocupacoes
    mean_amplitude_rms_cnn=sum_amplitude_rms_cnn/total_ocupacoes
    mean_amplitude_std_of=sum_amplitude_std_of/total_ocupacoes
    mean_amplitude_std_cnn=sum_amplitude_std_cnn/total_ocupacoes
    # FASE
    mean_fase_rms_of=sum_fase_rms_of/total_ocupacoes
    mean_fase_rms_cnn=sum_fase_rms_cnn/total_ocupacoes
    mean_fase_rms_cnn_estimated=sum_fase_rms_cnn_estimated/total_ocupacoes

    mean_fase_std_of= sum_fase_std_of/total_ocupacoes
    mean_fase_std_cnn=sum_fase_std_cnn/total_ocupacoes
    mean_fase_std_cnn_estimated=sum_fase_std_cnn_estimated/total_ocupacoes



    melhoria_amplitude_rms = ((mean_amplitude_rms_of-mean_amplitude_rms_cnn)/mean_amplitude_rms_of)*100
    melhoria_amplitude_std = ((mean_amplitude_std_of-mean_amplitude_std_cnn)/mean_amplitude_std_of)*100
    melhoria_fase_rms = ((mean_fase_rms_of-mean_fase_rms_cnn)/mean_fase_rms_of)*100
    melhoria_fase_std = ((mean_fase_std_of-mean_fase_std_cnn)/mean_fase_std_of)*100

    melhoria_fase_rms_cnn_estimated = ((mean_fase_rms_of-mean_fase_rms_cnn_estimated)/mean_fase_rms_of)*100
    melhoria_fase_std_cnn_estimated = ((mean_fase_std_of-mean_fase_std_cnn_estimated)/mean_fase_std_of)*100

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
    print(f"---------------------------- Amplitude estimada CNN --------------------------")
    print(f"Melhoria CNN vs OF - Fase RMS:      {melhoria_fase_rms:.5f}%")
    print(f"Melhoria CNN vs OF - Fase STD:      {melhoria_fase_std:.5f}%")
    print(f"---------------------------- Fase estimada CNN --------------------------")
    print(f"Melhoria CNN vs OF - Fase RMS:      {melhoria_fase_rms_cnn_estimated:.5f}%")
    print(f"Melhoria CNN vs OF - Fase STD:      {melhoria_fase_std_cnn_estimated:.5f}%")

def MeanStdAmplitude():
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
    cnn8_data_parcial = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_8")

    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"results_occupation_{ocupacao}.npz")
        cnn8_data_path = os.path.join(cnn8_data_parcial, f'results_ocupacao_{ocupacao}.npz')
        OF_data = np.load(OF_data_path)
        cnn8_data = np.load(cnn8_data_path)
        _, CNN_data_amplitude, _, _, cnn_type = DefinePath_CNN(ocupacao)
        print(f"{30*'-'} AMPLITUDE {30*'-'}\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print("| Metrica       | Mean      | Std       |")
        print("|" + "-"*15 + "|" + "-"*10 + "|")
        print(f"| OF            | {OF_data['std_error']:.6f} | {OF_data['std_std_error']:.6f} |")
        print(f"| {cnn_type}           | {CNN_data_amplitude['std_error']:.6f} | {CNN_data_amplitude['std_std_error']:.6f} |")
        print(f"| CNN-8           | {cnn8_data['std_error']:.6f} | {cnn8_data['std_std_error']:.6f} |")
        print(100*"=")

# ImprimeMetricas_Amplitude()

# ImprimeMetricas_Fase()
# MelhoriasCNN()

# MeanStdAmplitude()

import mplhep as hep
from collections import defaultdict

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
hep.style.use("ATLAS")


def PlotTableComparativeAmplitude(type=None):
    fontSize= 24
    ocupacoes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    if type == "Amplitude":
        chaves = ['OF', 'CNN3', 'CNN5']
    elif type == "Fase":
        chaves = ['OF', 'CNN3', 'CNN5', 'Real_Amp', 'CNN3_Tau', 'CNN5_Tau']
    else:
        raise ValueError("Tipo inválido. Escolha 'Amplitude' ou 'Fase'.")

    rms = {k: [] for k in chaves}
    r2 = {k: [] for k in chaves}
    mae = {k: [] for k in chaves}
    medae = {k: [] for k in chaves}

    if type == "Amplitude":
        estilos = {
                'OF':          {'cor': '#9900ff', 'marker': 'o', 'label': 'OF'},
                'CNN3':        {'cor': '#B0B0B0', 'marker': '*', 'label': 'CNN-3'},
                'CNN5':        {'cor': '#1A1A1A', 'marker': 's', 'label': 'CNN-5'},
                'CNN8':        {'cor': '#006130', 'marker': '^', 'label': 'CNN-8'},
                
            }
    elif type == "Fase":
        estilos = {
                'OF':          {'cor': '#9900ff', 'marker': 'o', 'label': r'$\hat{A}_{OF}$', 'linestyle': 'solid', 'zorder':1},
                'CNN3':        {'cor': '#B0B0B0', 'marker': 's', 'label': r'$\hat{A}_{CNN3}$', 'linestyle': 'solid', 'zorder':3},
                'CNN5':        {'cor': '#1A1A1A', 'marker': '*', 'label': r'$\hat{A}_{CNN5}$', 'linestyle': 'dashed', 'zorder':4},
                # 'CNN8':        {'cor': '#006130', 'marker': '*', 'label': r'$\hat{A}_{CNN8}$'},
                'Real_Amp':    {'cor': '#FA3232', 'marker': 'o', 'label': r'$A_{RA}$', 'linestyle': 'solid', 'zorder':2},
                'CNN3_Tau':     {'cor': 'darkorange', 'marker': 's', 'label': r'$\tau_{CNN3}$', 'linestyle': 'solid', 'zorder':1},
                'CNN5_Tau':     {'cor': 'deepskyblue', 'marker': '*', 'label': r'$\tau_{CNN5}$', 'linestyle': 'dashed', 'zorder':2},
                # 'CNN8_Tau':     {'cor': 'deepskyblue', 'marker': '*', 'label': r'$\tau_{CNN8}$'}
            }


    for ocupacao in ocupacoes:
        if type == "Amplitude":
            cnn8_data_parcial = os.path.join(base_path, "RedeNeuralConvolucional", "CNN_8")
            cnn8_data_path = os.path.join(cnn8_data_parcial, f'results_ocupacao_{ocupacao}.npz')
            CNN8_data = np.load(cnn8_data_path)

            of_data_parcial = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
            OF_data = np.load(os.path.join(of_data_parcial, f"results_occupation_{ocupacao}.npz"))

            _, CNN3_data, _, _, _ = DefinePath_CNN(ocupacao, 3)
            _, CNN5_data, _, _, _ = DefinePath_CNN(ocupacao, 5)

            dados_iteracao = {
                'OF': OF_data,
                'CNN3': CNN3_data,
                'CNN5': CNN5_data,
                'CNN8': CNN8_data,
            }
        elif type == "Fase":
            of_data_parcial = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_OF", f'janelamento_{n_janelamento}')
            real_amplitude_data_parcial = os.path.join(base_path, "FiltroOtimo", "FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}')
    
            OF_data_path = os.path.join(of_data_parcial, f"phase_of_occupation_{ocupacao}.npz")
            OF_data = np.load(OF_data_path)
            Real_Amplitude_data_path = os.path.join(real_amplitude_data_parcial, f"phase_real_amplitude_occupation_{ocupacao}.npz")
            Real_Amplitude_data = np.load(Real_Amplitude_data_path)

            _, _, CNN3_data_fase, CNN3_estimated_data_fase, cnn3_type = DefinePath_CNN(ocupacao, 3)
            _, _, CNN5_data_fase, CNN5_estimated_data_fase, cnn5_type = DefinePath_CNN(ocupacao, 5)
            _, _, CNN8_data_fase, CNN8_estimated_data_fase, cnn8_type = DefinePath_CNN(ocupacao, 8)

            dados_iteracao = {
                'OF': OF_data,
                'CNN3':CNN3_data_fase,     
                'CNN5': CNN5_data_fase,
                'Real_Amp': Real_Amplitude_data,
                'CNN3_Tau':  CNN3_estimated_data_fase,
                'CNN5_Tau': CNN5_estimated_data_fase,
            }

        for k in chaves:
            rms[k].append(dados_iteracao[k]['rms'])
            r2[k].append(dados_iteracao[k]['r2'])
            mae[k].append(dados_iteracao[k]['mae'])
            medae[k].append(dados_iteracao[k]['medae'])

    # fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    def plotar_metrica(ax, metrica_dict, titulo, ylabel):
        for k in chaves:
            ax.plot(ocupacoes, metrica_dict[k], color=estilos[k]['cor'], 
                    marker=estilos[k]['marker'], linestyle=estilos[k]['linestyle'], zorder=estilos[k]['zorder'], markersize=8)

        ax.set_title(titulo, fontsize=fontSize-1, fontweight='bold')
        ax.set_xlabel("Ocupação (%)", fontsize=fontSize-2)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.set_ylabel(ylabel, fontsize=fontSize-2)
        ax.set_xticks(ocupacoes)
        ax.grid(True, linestyle='--', alpha=0.6)
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((0, 0))
        # formatter.set_useOffset(True)
        # ax.yaxis.set_major_formatter(formatter)

    handles = []
    labels = []
    for k in chaves:
        handles.append(plt.Line2D([0], [0], color=estilos[k]['cor'], marker=estilos[k]['marker'], 
                                  linestyle=estilos[k]['linestyle'], linewidth=2, markersize=10))
        labels.append(estilos[k]['label'])

    if type=="Amplitude":
        unidade = 'ADC Counts'
    elif type=='Fase':
        unidade = 'ns'

    plotar_metrica(axs[0], rms, "RMS (Raiz do Erro Quadrático Médio)", f'{unidade}')
    plotar_metrica(axs[1], r2, "R² (Coeficiente de Determinação)", f'\u2013')
    plotar_metrica(axs[2], mae, "MAE (Erro Absoluto Médio)", f'{unidade}')
    plotar_metrica(axs[3], medae, "MedAE (Erro Absoluto Mediano)", f'{unidade}')

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        fontsize=fontSize-1
    )
    if type == "Fase":
        axs[1].set_ylim(0, 1.0)

    # fig.suptitle(f"Análise de Desempenho de Reconstrução da {type}", fontsize=16, fontweight='bold', y=0.96)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()
        
        
PlotTableComparativeAmplitude(type='Fase')
