import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7

base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

plt.rcParams['savefig.directory'] = os.path.dirname(path)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def PlotFaseDispersionOcupacao():
    of_dispersions = []
    cnn3_dispersions = []
    cnn5_dispersions = []
    real_amplitude_dispersions = []
    cnn3_estimated_dispersions = []
    cnn5_estimated_dispersions = []
    fontSize = 18
    for ocupacao in ocupacoes:
        # PATHS
        # TEM QUE PEGAR DA PASTA DO FILTRO OTIMO..;
        of_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_OF', f'janelamento_{n_janelamento}', f'phase_of_occupation_{ocupacao}.npz')
        cnn3_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_3', f'phase_cnn_occupation_{ocupacao}.npz' )
        cnn5_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_5', f'phase_cnn_occupation_{ocupacao}.npz' )
        real_amplitude_path = os.path.join(dataset_path, "FiltroOtimo", f"FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}', f'phase_real_amplitude_occupation_{ocupacao}.npz' )
        cnn3_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_3', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')
        cnn5_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_5', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')

        # LOADS
        of_load = np.load(of_path)
        cnn3_load = np.load(cnn3_path)
        cnn5_load = np.load(cnn5_path)
        real_amplitude_load = np.load(real_amplitude_path)
        cnn3_estimated_load = np.load(cnn3_estimated_path)
        cnn5_estimated_load = np.load(cnn5_estimated_path)

        # DATA LOAD
        data_of = of_load['std']
        data_cnn3 = cnn3_load['std']
        data_cnn5 = cnn5_load['std']
        data_real_amplitude = real_amplitude_load['std']
        data_cnn3_estimate = cnn3_estimated_load['std_error']
        data_cnn5_estimate = cnn5_estimated_load['std_error']

        of_dispersions.append(data_of)
        cnn3_dispersions.append(data_cnn3)
        cnn5_dispersions.append(data_cnn5)
        real_amplitude_dispersions.append(data_real_amplitude)
        cnn3_estimated_dispersions.append(data_cnn3_estimate)
        cnn5_estimated_dispersions.append(data_cnn5_estimate)

    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"
    cnn3_color = "#B0B0B0"
    cnn5_color = "#1A1A1A"
    cnn3_estimated_color = 'darkorange'
    cnn5_estimated_color = 'deepskyblue'
    
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(ocupacoes,of_dispersions, label="OF", marker='o', color = of_color)
    ax.plot(ocupacoes,cnn3_dispersions, label=r'CNN-3*', marker='*',linestyle='dashed', color = cnn3_color, zorder=5)
    ax.plot(ocupacoes,cnn5_dispersions, label=r'CNN-5*', marker='s', color = cnn5_color, linewidth=2)
    ax.plot(ocupacoes, real_amplitude_dispersions, label="Amplitude real", marker='o', color = real_amplitude_color)
    ax.plot(ocupacoes, cnn3_estimated_dispersions, label="CNN-3", marker='s', markersize=6, color=cnn3_estimated_color, linewidth=2, zorder=4)
    ax.plot(ocupacoes, cnn5_estimated_dispersions, label="CNN-5", marker='*', markersize=6, color=cnn5_estimated_color, linestyle='dashed', linewidth=1.5, zorder=5)
    ax.legend(loc='best')
    ax.set_xlabel('Ocupação (%)', fontsize=fontSize-2)
    ax.set_ylabel('Dispersão da fase estimada (ns)', fontsize=fontSize-2)
    ax.set_title(r'Dispersão $\times$ Ocupação', fontsize=fontSize-1)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # ZOOM 1 CNN
    axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.75, 0.25, 0.15, 0.15),  # (x, y) posição do canto
                   bbox_transform=ax.transAxes,   # coordenadas relativas ao gráfico
                   loc='center')
    
    x1,x2= 79.985, 80.010
    y1,y2 = 2.4035, 2.405
    axins.set_xticks([x1,x2])
    axins.set_yticks([y1,y2])

    axins.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
    axins.tick_params(axis='both', colors="#424242")

    axins.plot(ocupacoes, cnn3_estimated_dispersions, marker='s', markersize=6, color=cnn3_estimated_color, linewidth=2, zorder=4)
    axins.plot(ocupacoes, cnn5_estimated_dispersions, marker='*', markersize=6, color=cnn5_estimated_color, linestyle='dashed', linewidth=1.5, zorder=5)
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    # Aplica o tamanho DEPOIS de definir os ticks
    plt.setp(axins.get_xticklabels(which='both'), fontsize=8)
    plt.setp(axins.get_yticklabels(), fontsize=8)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray", linewidth=1.5)

    # ZOOM 2 CNN*
    axins1 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.1, 0.75, 0.15, 0.15), bbox_transform=ax.transAxes, loc='center')
    x11,x21= 9.9,10.1
    y11,y21=67.50,68.80
    axins1.set_xticks([x11,x21])
    axins1.set_yticks([y11,y21])

    axins1.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True,    labeltop=True)
    axins1.tick_params(axis='both', colors="#424242")

    axins1.plot(ocupacoes,cnn3_dispersions, marker='*',linestyle='dashed', color = cnn3_color, zorder=5)
    axins1.plot(ocupacoes,cnn5_dispersions, marker='s', color = cnn5_color, linewidth=2)
    axins1.plot(ocupacoes, real_amplitude_dispersions, marker='o', color = real_amplitude_color)
    axins1.set_xlim(x11,x21)
    axins1.set_ylim(y11,y21)
    # Aplica o tamanho DEPOIS de definir os ticks
    plt.setp(axins1.get_xticklabels(which='both'), fontsize=8)
    plt.setp(axins1.get_yticklabels(), fontsize=8)
    mark_inset(ax, axins1, loc1=3, loc2=4, fc="none", ec="gray", linewidth=1.5)

    plt.tight_layout()
    plt.show()


PlotFaseDispersionOcupacao()