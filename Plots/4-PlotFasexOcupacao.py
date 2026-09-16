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
cnn8_color ="#0BBE65"
real_amplitude_color = "#FA3232"
cnn3_estimated_color = 'darkorange'
cnn5_estimated_color = 'deepskyblue'
cnn8_estimated_color = 'magenta'


def LoadData(metric):
    """Função auxiliar para carregar os dados e evitar repetição"""
    of_metric, cnn3_metric, cnn5_metric, cnn8_metric, real_amplitude_metric, cnn3_estimated_metric, cnn5_estimated_metric, cnn8_estimated_metric = [], [], [], [], [], [], [] , []
    of_err, cnn3_err, cnn5_err, cnn8_err, real_amplitude_err,  cnn3_estimated_err, cnn5_estimated_err, cnn8_estimated_err = [], [], [], [], [] , [] , [] , []

    for ocupacao in ocupacoes:
        # TEM QUE PEGAR DA PASTA DO FILTRO OTIMO..;
        of_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_OF', f'janelamento_{n_janelamento}', f'phase_of_occupation_{ocupacao}.npz')
        cnn3_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_3', f'phase_cnn_occupation_{ocupacao}.npz' )
        cnn5_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_5', f'phase_cnn_occupation_{ocupacao}.npz' )
        cnn8_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_8', f'phase_cnn_occupation_{ocupacao}.npz' )
        real_amplitude_path = os.path.join(dataset_path, "FiltroOtimo", f"FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}', f'phase_real_amplitude_occupation_{ocupacao}.npz' )
        cnn3_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_3', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')
        cnn5_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_5', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')
        cnn8_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_8', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')

        # LOADS
        of_load = np.load(of_path)
        cnn3_load = np.load(cnn3_path)
        cnn5_load = np.load(cnn5_path)
        cnn8_load = np.load(cnn8_path)
        real_amplitude_load = np.load(real_amplitude_path)
        cnn3_estimated_load = np.load(cnn3_estimated_path)
        cnn5_estimated_load = np.load(cnn5_estimated_path)
        cnn8_estimated_load = np.load(cnn8_estimated_path)
        
        of_load = np.load(of_path)
        cnn3_load = np.load(cnn3_path)
        cnn5_load = np.load(cnn5_path)
        cnn8_load = np.load(cnn8_path)
        if metric== "mean":
            of_metric.append(of_load['mean'])
            cnn3_metric.append(cnn3_load['mean'])
            cnn5_metric.append(cnn5_load['mean'])
            cnn8_metric.append(cnn8_load['mean'])
            real_amplitude_metric.append(real_amplitude_load['mean'])

            cnn3_estimated_metric.append(cnn3_estimated_load['mean_error'])
            cnn5_estimated_metric.append(cnn5_estimated_load['mean_error'])
            cnn8_estimated_metric.append(cnn8_estimated_load['mean_error'])
            
            '''
                OS MÉTODOS QUE DIVIDEM POR AMPLITUDE ESTIMADA NAO TEM DESVIO PADRAO POR FOLD
            '''
            # cnn3_estimated_err.append(cnn3_estimated_load['std_mean_error'])
            # cnn5_estimated_err.append(cnn5_estimated_load['std_mean_error'])
            # cnn8_estimated_err.append(cnn8_estimated_load['std_mean_error'])


        elif metric=="std":
            of_metric.append(of_load['std'])
            cnn3_metric.append(cnn3_load['std'])
            cnn5_metric.append(cnn5_load['std'])
            cnn8_metric.append(cnn8_load['std'])
            real_amplitude_metric.append(real_amplitude_load['std'])
            cnn3_estimated_metric.append(cnn3_estimated_load['std_mean_error'])
            cnn5_estimated_metric.append(cnn5_estimated_load['std_mean_error'])
            cnn8_estimated_metric.append(cnn8_estimated_load['std_mean_error'])
            
            '''
                OS MÉTODOS QUE DIVIDEM POR AMPLITUDE ESTIMADA NAO TEM DESVIO PADRAO POR FOLD
            '''
            # cnn3_estimated_err.append(cnn3_estimated_load['std_std_error'])
            # cnn5_estimated_err.append(cnn5_estimated_load['std_std_error'])
            # cnn8_estimated_err.append(cnn8_estimated_load['std_std_error'])

        
    return (of_metric, cnn3_metric, cnn5_metric, cnn8_metric, real_amplitude_metric, cnn3_estimated_metric, cnn5_estimated_metric, cnn8_estimated_metric), (of_err, cnn3_err, cnn5_err, cnn8_err, real_amplitude_err,  cnn3_estimated_err, cnn5_estimated_err, cnn8_estimated_err)


def PlotFaseDispersionOcupacao(metric, cnn8, zoom):
    (of_metric, cnn3_metric, cnn5_metric, cnn8_metric, real_amplitude_metric, cnn3_estimated_metric, cnn5_estimated_metric, cnn8_estimated_metric), (of_err, cnn3_err, cnn5_err, cnn8_err, real_amplitude_err,  cnn3_estimated_err, cnn5_estimated_err, cnn8_estimated_err) = LoadData(metric)

    fontSize = 18

    fig, ax = plt.subplots(figsize=(7, 5))

    if metric=='mean':
        y_label = r'$\mu$ (ns)'
        ax.set_ylim(-12,1)

    elif metric=='std':
        y_label = r'$\sigma$ (ns)'

        if zoom:
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

            axins.plot(ocupacoes, cnn3_estimated_metric, marker='s', markersize=6, color=cnn3_estimated_color, linewidth=2, zorder=4)
            axins.plot(ocupacoes, cnn5_estimated_metric, marker='*', markersize=6, color=cnn5_estimated_color, linestyle='dashed', linewidth=1.5, zorder=5)
            axins.set_xlim(x1,x2)
            axins.set_ylim(y1,y2)
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

            axins1.plot(ocupacoes, real_amplitude_metric, marker='o', color = real_amplitude_color, zorder=2)
            axins1.plot(ocupacoes,cnn3_metric, marker='*',linestyle='dashed', color = cnn3_color, zorder=4)
            axins1.plot(ocupacoes,cnn5_metric, marker='s', color = cnn5_color, linewidth=2, zorder=1)
            if cnn8:
                axins1.plot(ocupacoes,cnn8_metric, marker='s', color = cnn5_color, linewidth=2, zorder=6)
            axins1.set_xlim(x11,x21)
            axins1.set_ylim(y11,y21)
            plt.setp(axins1.get_xticklabels(which='both'), fontsize=8)
            plt.setp(axins1.get_yticklabels(), fontsize=8)
            mark_inset(ax, axins1, loc1=3, loc2=4, fc="none", ec="gray", linewidth=1.5)

    ax.set_xlim(-2,102)
    ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
    of_err = np.array(of_err)
    cnn3_err = np.array(cnn3_err)
    cnn5_err = np.array(cnn5_err)
    cnn8_err = np.array(cnn8_err)

    ax.plot(ocupacoes, of_metric,  label=r'$\hat{A}_{OF}$', marker='o', linestyle='-', color=of_color, markersize=6)
    ax.plot(ocupacoes, real_amplitude_metric,  label=r'$A_{RA}$', marker='o', linestyle='-', color=real_amplitude_color, markersize=6)
    ax.plot(ocupacoes, cnn3_metric, label=r'$\hat{A}_{CNN3}$', marker='*', linestyle='dashed', color=cnn3_color, zorder=5, markersize=6)
    ax.plot(ocupacoes, cnn5_metric, label=r'$\hat{A}_{CNN5}$', marker='s', linestyle='-', color=cnn5_color, markersize=6)
    if cnn8:
        ax.plot(ocupacoes, cnn8_metric, label=r'$\hat{A}_{CNN8}$', marker='^', linestyle='-.', color=cnn8_color,zorder=6, markersize=4)

    ax.plot(ocupacoes, cnn3_estimated_metric, label=r'$\tau_{CNN3}$', linestyle='dashed', markersize=6, color=cnn3_estimated_color, marker='*',zorder=2)
    ax.plot(ocupacoes, cnn5_estimated_metric, label=r'$\tau_{CNN5}$', linestyle='solid', markersize=6, color=cnn5_estimated_color, marker='s',zorder=1)

    if cnn8:
        ax.plot(ocupacoes, cnn8_estimated_metric, label=r'$\tau_{CNN8}$', marker='^', linestyle='-.', color=cnn8_estimated_color, zorder=6, markersize=4)

    ax.legend(loc='best')
    ax.set_xlabel('Ocupação (%)', fontsize=fontSize-2)
    ax.set_ylabel(y_label, fontsize=fontSize-2)
    # ax.set_title(r'Dispersão $\times$ Ocupação', fontsize=fontSize-1)
    ax.tick_params(axis='both', which='major', labelsize=14)
   
    plt.tight_layout()
    plt.show()


PlotFaseDispersionOcupacao(metric='std', cnn8=True, zoom=True)