import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import mplhep as hep

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7

base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

plt.rcParams['savefig.directory'] = os.path.dirname(path)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
hep.style.use("ATLAS")


of_color = '#9900ff'
cnn3_color = "#B0B0B0"
cnn5_color = "#1A1A1A"
cnn8_color = "#006130"

def LoadData():
    """Função auxiliar para carregar os dados e evitar repetição"""
    of_disp, cnn3_disp, cnn5_disp, cnn8_disp = [], [], [], []
    of_err, cnn3_err, cnn5_err, cnn8_err = [], [], [], []

    for ocupacao in ocupacoes:
        of_path = os.path.join(dataset_path, "FiltroOtimo", f'AmplitudeEstimada_OF', f'janelamento_{n_janelamento}', f'results_occupation_{ocupacao}.npz')
        cnn3_path = os.path.join(dataset_path, "RedeNeuralConvolucional", f'CNN_3', f'results_ocupacao_{ocupacao}.npz')
        cnn5_path = os.path.join(dataset_path, "RedeNeuralConvolucional", f'CNN_5', f'results_ocupacao_{ocupacao}.npz')
        cnn8_path = os.path.join(dataset_path, "RedeNeuralConvolucional", f'CNN_8', f'results_ocupacao_{ocupacao}.npz')
        
        of_load = np.load(of_path)
        cnn3_load = np.load(cnn3_path)
        cnn5_load = np.load(cnn5_path)
        cnn8_load = np.load(cnn8_path)
        
        # Dispersões
        of_disp.append(of_load['std_error'])
        cnn3_disp.append(cnn3_load['std_error'])
        cnn5_disp.append(cnn5_load['std_error'])
        cnn8_disp.append(cnn8_load['std_error'])

        # Erros (std_std_error)
        of_err.append(of_load['std_std_error'])
        cnn3_err.append(cnn3_load['std_std_error'])
        cnn5_err.append(cnn5_load['std_std_error'])
        cnn8_err.append(cnn8_load['std_std_error'])
        
    return (of_disp, cnn3_disp, cnn5_disp, cnn8_disp), (of_err, cnn3_err, cnn5_err, cnn8_err)

def PlotAmplitudeDispersionOcupacao_Zoom():
    (of_disp, cnn3_disp, cnn5_disp, cnn8_disp), (of_err, cnn3_err, cnn5_err, cnn8_err) = LoadData()
    fontSize = 18
    
    fig, ax = plt.subplots(figsize=(7, 5))

    # Trocado ax.plot por ax.errorbar
    ax.errorbar(ocupacoes, of_disp, yerr=of_err, label="OF", marker='o', linestyle='-', color=of_color, capsize=3)
    ax.errorbar(ocupacoes, cnn3_disp, yerr=cnn3_err, label=r'CNN-3', marker='*', linestyle='dashed', color=cnn3_color, zorder=5, capsize=3)
    ax.errorbar(ocupacoes, cnn5_disp, yerr=cnn5_err, label=r'CNN-5', marker='s', linestyle='-', color=cnn5_color, linewidth=2, capsize=3)

    ax.legend(loc='best')
    ax.set_xlabel('Ocupação (%)', fontsize=fontSize-2)
    ax.set_ylabel('Dispersão da amplitude\nestimada (ADC Counts)', fontsize=fontSize-2)
    ax.set_title(r'Dispersão $\times$ Ocupação', fontsize=fontSize-1)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # ZOOM 3 CNN (Comentado a pedido)
    '''
    axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.75, 0.25, 0.1, 0.1), bbox_transform=ax.transAxes, loc='center')
    x1, x2 = 79.985, 80.010
    y1, y2 = 20.87, 20.96
    axins.set_xticks([x1, x2])
    axins.set_yticks([y1, y2])
    axins.errorbar(ocupacoes, cnn3_disp, yerr=cnn3_err, marker='*', linestyle='dashed', color=cnn3_color, zorder=5)
    axins.errorbar(ocupacoes, cnn5_disp, yerr=cnn5_err, marker='s', linestyle='-', color=cnn5_color, linewidth=2)
    axins.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
    axins.tick_params(axis='both', colors="black")
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.setp(axins.get_xticklabels(which='both'), fontsize=8)
    plt.setp(axins.get_yticklabels(), fontsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black", linewidth=1.5)
    '''

    # ZOOM 1 CNN (Comentado a pedido)
    '''
    axins1 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.25, 0.3, 0.1, 0.1), bbox_transform=ax.transAxes, loc='center')
    x11, x21 = 29.985, 30.010
    y11, y21 = 7.4, 7.7
    axins1.set_xticks([x11, x21])
    axins1.set_yticks([y11, y21])
    axins1.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
    axins1.tick_params(axis='both', colors="#424242")
    axins1.errorbar(ocupacoes, cnn3_disp, yerr=cnn3_err, marker='*', linestyle='dashed', color=cnn3_color, zorder=5)
    axins1.errorbar(ocupacoes, cnn5_disp, yerr=cnn5_err, marker='s', linestyle='-', color=cnn5_color, linewidth=2)
    axins1.set_xlim(x11, x21)
    axins1.set_ylim(y11, y21)
    plt.setp(axins1.get_xticklabels(which='both'), fontsize=8)
    plt.setp(axins1.get_yticklabels(), fontsize=8)
    mark_inset(ax, axins1, loc1=3, loc2=4, fc="none", ec="black", linewidth=1.5)
    '''

    # ZOOM 2 CNN (Mantido)
    axins2 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.5, 0.45, 0.2, 0.2), bbox_transform=ax.transAxes, loc='center')
    x11, x21 = 59.8, 70.20
    y11, y21 = 15.9, 19
    axins2.set_xticks([x11, x21])
    axins2.set_yticks([y11, y21])
    axins2.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
    axins2.tick_params(axis='both', colors="#424242")
    
    # Aplicando errorbar no zoom também
    axins2.errorbar(ocupacoes, cnn3_disp, yerr=cnn3_err, marker='*', linestyle='dashed', color=cnn3_color, zorder=5, capsize=2)
    axins2.errorbar(ocupacoes, cnn5_disp, yerr=cnn5_err, marker='s', linestyle='-', color=cnn5_color, linewidth=2, capsize=2)
    
    axins2.set_xlim(x11, x21)
    axins2.set_ylim(y11, y21)
    plt.setp(axins2.get_xticklabels(which='both'), fontsize=8)
    plt.setp(axins2.get_yticklabels(), fontsize=8)
    mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="black", linewidth=1.5)

    plt.tight_layout()
    plt.show()


#PlotAmplitudeDispersionOcupacao_Zoom()

def PlotAmplitudeDispersionOcupacao_Subplot():
    (of_disp, cnn3_disp, cnn5_disp, cnn8_disp), (of_err, cnn3_err, cnn5_err, cnn8_err) = LoadData()
    fontSize = 18
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [2, 1.2]})
    fig.subplots_adjust(hspace=0.3) 

    # ==========================================
    # PAINEL PRINCIPAL (Cima) - Visão de 0 a 100
    # ==========================================
    ax1.errorbar(ocupacoes, of_disp, yerr=of_err, label="OF", marker='o', linestyle='-', color=of_color, capsize=3)
    ax1.errorbar(ocupacoes, cnn3_disp, yerr=cnn3_err, label=r'CNN-3', marker='*', linestyle='dashed', color=cnn3_color, zorder=5, capsize=3)
    ax1.errorbar(ocupacoes, cnn5_disp, yerr=cnn5_err, label=r'CNN-5', marker='s', linestyle='-', color=cnn5_color, linewidth=2, capsize=3)

    ax1.legend(loc='upper left')
    ax1.set_ylabel('Dispersão da amplitude\nestimada (ADC Counts)', fontsize=fontSize-4)
    ax1.set_title(r'Dispersão $\times$ Ocupação', fontsize=fontSize-1)
    ax1.set_xlabel('Ocupação (%)', fontsize=fontSize-2)
    ax1.set_xlim(-2, 102) 
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # ==========================================
    # PAINEL ZOOM (Baixo) - Foco no cruzamento
    # ==========================================
    ax2.errorbar(ocupacoes, cnn3_disp, yerr=cnn3_err, label=r'CNN-3', marker='*', linestyle='dashed', color=cnn3_color, zorder=5, capsize=3, markersize=10)
    ax2.errorbar(ocupacoes, cnn5_disp, yerr=cnn5_err, label=r'CNN-5', marker='s', linestyle='-', color=cnn5_color, linewidth=2, capsize=3, markersize=7)
    
    ax2.set_xlabel('Ocupação (%)', fontsize=fontSize-2)
    ax2.set_ylabel('Zoom (ADC)', fontsize=fontSize-6) # Deixa claro que é um recorte
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    ax2.set_xlim(58, 72)
    ax2.set_ylim(15, 20)
    ax2.set_xticks([60, 70])
    
    ax2.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.show()

PlotAmplitudeDispersionOcupacao_Subplot()