import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import mplhep as hep
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

n_janelamento = 7
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
hep.style.use("ATLAS")

plt.rcParams['savefig.directory'] = os.path.dirname(path)

def SaveDataMeanSTD_CNN(metric):
    occupation_new = [0,10,20,30,40,50,60,70,80,90,100]
    CNN_paths = [3,5,8]
    base_path = os.path.dirname(os.path.dirname(path))
    
    data_cnn_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks","RedeNeuralConvolucional")
    CNN_Data = {}
    for cnn in CNN_paths:
        CNN_path =  os.path.join(data_cnn_path, f"CNN_{cnn}")
        for occupation in occupation_new:
            CNN_path_complete = os.path.join(CNN_path, f"results_ocupacao_{occupation}.npz")
            CNN_data = np.load(CNN_path_complete)
            #{(CNN architecture, occupation): (mean error, std error)}
            metric = metric.lower()
            if metric=='mean':
                CNN_Data[(cnn, occupation)] = (CNN_data['mean_error'], CNN_data['std_mean_error'])
            elif metric=='std':
                CNN_Data[(cnn, occupation)] = (CNN_data['std_error'], CNN_data['std_std_error'])

    return CNN_Data

def SaveDataMeanSTD_OF(metric):
    occupation_new = [0,10,20,30,40,50,60,70,80,90,100]
    base_path = os.path.dirname(os.path.dirname(path))
    
    of_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks", "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_7')
    OF_Data = {}
    for occupation in occupation_new:
        of_file = os.path.join(of_path, f"results_occupation_{occupation}.npz")
        of_data = np.load(of_file)
        metric_lower = metric.lower()
        if metric_lower == 'mean':
            OF_Data[occupation] = (of_data['mean_error'], of_data['std_mean_error'])
        elif metric_lower == 'std':
            OF_Data[occupation] = (of_data['std_error'], of_data['std_std_error'])
    
    return OF_Data

def GetMeanCNN(CNN_Data, cnn_target):
    filtrado = {
        occ: (mean, std)
        for (cnn, occ), (mean, std) in CNN_Data.items()
        if cnn == cnn_target
    }

    occupations = list(filtrado.keys())
    means = [filtrado[o][0] for o in occupations]
    stds  = [filtrado[o][1] for o in occupations]

    return occupations, means, stds

def GetMeanOF(OF_Data):
    occupations = list(OF_Data.keys())
    means = [OF_Data[o][0] for o in occupations]
    stds  = [OF_Data[o][1] for o in occupations]
    return occupations, means, stds

def Plot_CNNxOF(metric, zoom):
    CNN_Data = SaveDataMeanSTD_CNN(metric)
    OF_Data = SaveDataMeanSTD_OF(metric)
   
    occupations_CNN3, means_CNN3, stds_CNN3 = GetMeanCNN(CNN_Data, 3)
    occupations_CNN5, means_CNN5, stds_CNN5 = GetMeanCNN(CNN_Data, 5)
    occupations_OF, means_OF, stds_OF = GetMeanOF(OF_Data)

    total_inches_image = 6.32
    fontSize = 24
    of_color = '#9900ff'

    fig, ax = plt.subplots(2, 1, figsize=(total_inches_image, 4), constrained_layout=True)
    ax = ax.flatten()

    x = occupations_OF
    y = means_OF
    yerr = stds_OF
    
    if metric=='mean':
        # y_label = 'Mean values\n(ADC counts)'
        y_label = r'$\bar{\mu}$ (ADC counts)'
        if zoom:
            #Plot de Cima
            #CNN mean
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins1 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.5, 0.75, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-4)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-4)+5*10
            y_inf_limite_cnn_mean = -0.3
            y_sup_limite_cnn_mean = 0.8

            axins1.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins1.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins1.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins1.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True,    labeltop=True)
            axins1.tick_params(axis='both', colors="#1A1A1A")

            axins1.errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="#1A1A1A", label='CNN-5', zorder=1)
            
            axins1.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins1.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins1.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins1.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins1, loc1=3, loc2=4, fc="none", ec="black", linewidth=1.5)

            #OF mean values
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins2 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.5, 0.25, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = -50.4
            y_sup_limite_of_mean = -49.6

            axins2.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins2.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins2.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins2.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
            axins2.tick_params(axis='both', colors="#1A1A1A")

            axins2.errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins2.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins2.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins2.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins2.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins2, loc1=1, loc2=2, fc="none", ec="purple", linewidth=1.5)

            #Plot de baixo
            #CNN mean
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins11 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.5, 0.75, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-4)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-4)+5*10
            y_inf_limite_cnn_mean = -0.3
            y_sup_limite_cnn_mean = 1.0

            axins11.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins11.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins11.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins11.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True,    labeltop=True)
            axins11.tick_params(axis='both', colors="#1A1A1A")

            axins11.errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
            
            axins11.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins11.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins11.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins11.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins11, loc1=3, loc2=4, fc="none", ec="gray", linewidth=1.5)

            #OF mean values
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins21 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.5, 0.25, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = -50.4
            y_sup_limite_of_mean = -49.6

            axins21.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins21.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins21.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins21.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
            axins21.tick_params(axis='both', colors="#1A1A1A")

            axins21.errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins21.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins21.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins21.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins21.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins21, loc1=1, loc2=2, fc="none", ec="purple", linewidth=1.5)
            
    elif metric=='std':
        # y_label = 'Mean dispersion\nvalues (ADC counts)'
        y_label = r'$\bar{\sigma}$ (ADC counts)'
        if zoom:
            #Plot de Cima
            #CNN mean
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins1 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.5, 0.15, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-3)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-3)+5*10
            y_inf_limite_cnn_mean = 13.05
            y_sup_limite_cnn_mean = 13.5

            axins1.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins1.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins1.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins1.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False,    labeltop=False)
            axins1.tick_params(axis='both', colors="#1A1A1A")

            axins1.errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="#1A1A1A", label='CNN-5', zorder=1)
            
            axins1.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins1.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins1.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins1.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins1, loc1=1, loc2=2, fc="none", ec="black", linewidth=1.5)

            #OF mean values
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins2 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.5, 0.75, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = 40.8
            y_sup_limite_of_mean = 41.8

            axins2.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins2.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins2.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins2.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
            axins2.tick_params(axis='both', colors="#1A1A1A")

            axins2.errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins2.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins2.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins2.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins2.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins2, loc1=3, loc2=4, fc="none", ec="purple", linewidth=1.5)

            #Plot de baixo
            #CNN mean
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins11 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.5, 0.15, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-4)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-4)+5*10
            y_inf_limite_cnn_mean = 13.05
            y_sup_limite_cnn_mean = 13.6

            axins11.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins11.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins11.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins11.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False,    labeltop=False)
            axins11.tick_params(axis='both', colors="#1A1A1A")

            axins11.errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
            
            axins11.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins11.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins11.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins11.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins11, loc1=1, loc2=2, fc="none", ec="gray", linewidth=1.5)

            #OF mean values
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins21 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.5, 0.75, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = 40.8
            y_sup_limite_of_mean = 41.8

            axins21.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins21.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins21.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins21.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
            axins21.tick_params(axis='both', colors="#1A1A1A")

            axins21.errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins21.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins21.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins21.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins21.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins21, loc1=3, loc2=4, fc="none", ec="purple", linewidth=1.5)

    #Plot de cima
    ax[0].errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="#1A1A1A", label='CNN-5', zorder=1)
    ax[0].errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
    ax[0].set_xlabel("Ocupação (%)", fontsize= fontSize)
    ax[0].set_ylabel(y_label, fontsize= fontSize)
    ax[0].legend(loc='best')
    ax[0].tick_params(axis='both', which='major', labelsize=20)

    # for i, (xi, yi, err) in enumerate(zip(x, y, yerr)):
    #     ax[0].plot([xi, xi], [yi - err, yi + err], linestyle='None', color=of_color, linewidth=2)
    #     ax[0].plot([xi - 0.2, xi + 0.2], [yi - err, yi - err], color=of_color, linewidth=2)
    #     ax[0].plot([xi - 0.2, xi + 0.2], [yi + err, yi + err], color=of_color, linewidth=2)

    #Plot de baixo
    ax[1].errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
    ax[1].errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
    ax[1].set_xlabel("Ocupação (%)", fontsize= fontSize)
    ax[1].set_ylabel(y_label, fontsize= fontSize)
    ax[1].legend(loc='best')  
    ax[1].tick_params(axis='both', which='major', labelsize=20)

    #plt.tight_layout()
    plt.show()

def Plot_CNN(metric, zoom, box):
    CNN_Data = SaveDataMeanSTD_CNN(metric)

    OF_Data = SaveDataMeanSTD_OF(metric)
   
    occupations_OF, means_OF, stds_OF = GetMeanOF(OF_Data)
    occupations_CNN3, means_CNN3, stds_CNN3 = GetMeanCNN(CNN_Data, 3)
    occupations_CNN5, means_CNN5, stds_CNN5 = GetMeanCNN(CNN_Data, 5)
    occupations_CNN8, means_CNN8, stds_CNN8 = GetMeanCNN(CNN_Data, 8)
    total_inches_image = 6.32
    fontSize = 24
    cnn8_color ="#006130"
    of_color = '#9900ff'
    fig, ax = plt.subplots(2, 1, figsize=(total_inches_image, 4), constrained_layout=True)
    ax = ax.flatten()

    x = occupations_CNN8
    y = means_CNN8
    yerr = stds_CNN8


    if metric=='mean':
        y_label = r'$\bar{\mu}~(ADC\,\,counts)$'
        # y_label = 'Mean values\n(ADC counts)'

        if zoom:
            #Plot de Cima
            #CNN mean
            axins1 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.298, 0.75, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-3)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-3)+5*10
            y_inf_limite_cnn_mean = -0.3
            y_sup_limite_cnn_mean = 0.8

            axins1.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins1.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins1.ticklabel_format(axis='both', style='plain', useOffset=False)
            axins1.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)

            axins1.tick_params(axis='both', colors="#1A1A1A")

            axins1.errorbar(occupations_CNN5,means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="black", zorder=10)
            axins1.errorbar(x, y, yerr=yerr,
                fmt='*', color=cnn8_color, label='CNN-8',
                zorder=10,
                capsize=3,       # ← caps horizontais de volta
                                                    
                )  
            
            axins1.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins1.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins1.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins1.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins1, loc1=3, loc2=4, fc="none", ec="black", linewidth=1.5)
            #OF mean values
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins2 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.298, 0.25, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = -50.35
            y_sup_limite_of_mean = -49.7

            axins2.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins2.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins2.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins2.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
            axins2.tick_params(axis='both', colors="#1A1A1A")

            axins2.errorbar(occupations_OF,means_OF, yerr=stds_OF, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins2.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins2.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins2.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins2.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins2, loc1=1, loc2=2, fc="none", ec="purple", linewidth=1.5)

            #Plot de baixo
            #CNN mean
            axins11 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.298, 0.75, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-3)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-3)+5*10
            y_inf_limite_cnn_mean = -0.3
            y_sup_limite_cnn_mean = 0.9

            axins11.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins11.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins11.ticklabel_format(axis='both', style='plain', useOffset=False)
            axins11.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)

            axins11.tick_params(axis='both', colors="#1A1A1A")

            axins11.errorbar(occupations_CNN3,means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color="#B0B0B0", zorder=10)
            axins11.errorbar(x, y, yerr=yerr,
                fmt='*', color=cnn8_color, label='CNN-8',
                zorder=10,
                capsize=3,       # ← caps horizontais de volta
                                                    
                )  
            axins11.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins11.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins11.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins11.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins11, loc1=3, loc2=4, fc="none", ec="#B0B0B0", linewidth=1.5)
            #OF mean values
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins21 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.298, 0.25, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = -50.35
            y_sup_limite_of_mean = -49.7

            axins21.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins21.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins21.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins21.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
            axins21.tick_params(axis='both', colors="#1A1A1A")

            axins21.errorbar(occupations_OF,means_OF, yerr=stds_OF, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins21.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins21.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins21.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins21.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins21, loc1=1, loc2=2, fc="none", ec="purple", linewidth=1.5)

        if box:
            # Adicionando janelas de zoom
            #CIMA
            inset_axes_ax = inset_axes(
                ax[0], width="100%", height="100%",
                loc="upper right",
                bbox_to_anchor=(0.69, 0.17, 0.3, 0.8),
                bbox_transform=ax[0].transAxes
            )
            inset_axes_ax.tick_params(axis='both', colors="#333333")
            inset_axes_ax.xaxis.label.set_color('#333333')
            inset_axes_ax.yaxis.label.set_color('#333333')

            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(True)
            inset_axes_ax.yaxis.set_major_formatter(formatter)

            inset_axes_ax.errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="black", label='CNN-5', zorder=1)
            inset_axes_ax.errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3,       # ← caps horizontais de volta
                                      
                                      )  
            inset_axes_ax.legend(loc='lower left')
            x_inf_limite = -1
            x_sup_limite = 101
            y_inf_limite = -2.6
            y_sup_limite = 0.8
            inset_axes_ax.set_xlim([x_inf_limite, x_sup_limite])
            inset_axes_ax.set_ylim([y_inf_limite, y_sup_limite])

            #Baixo
            inset_axes_ax1 = inset_axes(
                ax[1], width="100%", height="100%",
                loc="upper right",
                bbox_to_anchor=(0.69, 0.17, 0.3, 0.8),
                bbox_transform=ax[1].transAxes
            )
            inset_axes_ax1.tick_params(axis='both', colors="#333333")
            inset_axes_ax1.xaxis.label.set_color('#333333')
            inset_axes_ax1.yaxis.label.set_color('#333333')

            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(True)
            inset_axes_ax1.yaxis.set_major_formatter(formatter)

            inset_axes_ax1.errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=1)
            inset_axes_ax1.errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3, 
                                      
                                      )  
            inset_axes_ax1.legend(loc='lower left')
            x_inf_limite = -1
            x_sup_limite = 101
            y_inf_limite = -2.6
            y_sup_limite = 0.95
            inset_axes_ax1.set_xlim([x_inf_limite, x_sup_limite])
            inset_axes_ax1.set_ylim([y_inf_limite, y_sup_limite])
            
    elif metric=='std':
        y_label = r'$\bar{\sigma}~(ADC\,\,counts)$'
        # y_label = 'Mean dispersion\nvalues (ADC counts)'
        if zoom:
            #Plot de cima
            #CNN mean
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins11 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.298, 0.15, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-4)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-4)+5*10
            y_inf_limite_cnn_mean = 12.9
            y_sup_limite_cnn_mean = 13.5

            axins11.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins11.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins11.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins11.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False,    labeltop=False)
            axins11.tick_params(axis='both', colors="#1A1A1A")

            axins11.errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color='#1A1A1A', label='CNN-3', zorder=0)
            axins11.errorbar(x, y, yerr=yerr,
                                        fmt='*', color=cnn8_color, label='CNN-8',
                                        zorder=10,
                                        capsize=3,       # ← caps horizontais de volta
                                        
                                        )    
            
            axins11.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins11.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins11.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins11.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins11, loc1=1, loc2=2, fc="none", ec="black", linewidth=1.5)

            #OF std
            axins2 = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(0.298, 0.75, 0.05, 0.15), bbox_transform=ax[0].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = 40.85
            y_sup_limite_of_mean = 41.8

            axins2.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins2.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins2.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins2.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
            axins2.tick_params(axis='both', colors="#1A1A1A")

            axins2.errorbar(occupations_OF,means_OF, yerr=stds_OF, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins2.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins2.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins2.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins2.get_yticklabels(), fontsize=11)
            mark_inset(ax[0], axins2, loc1=3, loc2=4, fc="none", ec="purple", linewidth=1.5)


            #Plot de baixo
            #CNN mean
            # bbox_to_anchor = (distancia da esquerda, distancia de baixo, largura, altura)
            axins11 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.298, 0.15, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_cnn_mean = -0.1*10**(-4)+5*10
            x_sup_limite_cnn_mean = 0.1*10**(-4)+5*10
            y_inf_limite_cnn_mean = 12.9
            y_sup_limite_cnn_mean = 13.62

            axins11.set_xticks([x_inf_limite_cnn_mean, x_sup_limite_cnn_mean])
            axins11.set_yticks([y_inf_limite_cnn_mean, y_sup_limite_cnn_mean])
            axins11.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins11.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False,    labeltop=False)
            axins11.tick_params(axis='both', colors="#1A1A1A")

            axins11.errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
            axins11.errorbar(x, y, yerr=yerr,
                                        fmt='*', color=cnn8_color, label='CNN-8',
                                        zorder=10,
                                        capsize=3,       # ← caps horizontais de volta
                                        
                                        )    
            
            axins11.set_xlim(x_inf_limite_cnn_mean, x_sup_limite_cnn_mean)
            axins11.set_ylim(y_inf_limite_cnn_mean, y_sup_limite_cnn_mean)

            plt.setp(axins11.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins11.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins11, loc1=1, loc2=2, fc="none", ec="gray", linewidth=1.5)

            #OF std
            axins21 = inset_axes(ax[1], width="100%", height="100%", bbox_to_anchor=(0.298, 0.75, 0.05, 0.15), bbox_transform=ax[1].transAxes, loc='center')
            x_inf_limite_of_mean = -0.1*10**(-3)+5*10
            x_sup_limite_of_mean = 0.1*10**(-3)+5*10
            y_inf_limite_of_mean = 40.85
            y_sup_limite_of_mean = 41.8

            axins21.set_xticks([x_inf_limite_of_mean, x_sup_limite_of_mean])
            axins21.set_yticks([y_inf_limite_of_mean, y_sup_limite_of_mean])
            axins21.ticklabel_format(axis='both', style='plain', useOffset=False)
            
            axins21.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
            axins21.tick_params(axis='both', colors="#1A1A1A")

            axins21.errorbar(occupations_OF,means_OF, yerr=stds_OF, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
            
            axins21.set_xlim(x_inf_limite_of_mean, x_sup_limite_of_mean)
            axins21.set_ylim(y_inf_limite_of_mean, y_sup_limite_of_mean)

            plt.setp(axins21.get_xticklabels(which='both'), fontsize=11)
            plt.setp(axins21.get_yticklabels(), fontsize=11)
            mark_inset(ax[1], axins21, loc1=3, loc2=4, fc="none", ec="purple", linewidth=1.5)

        if box:
            # Adicionando janelas de zoom
            #CIMA
            inset_axes_ax = inset_axes(
                ax[0], width="100%", height="100%",
                loc="upper right",
                bbox_to_anchor=(0.69, 0.17, 0.3, 0.8),
                bbox_transform=ax[0].transAxes
            )
            inset_axes_ax.tick_params(axis='both', colors="#333333")
            inset_axes_ax.xaxis.label.set_color('#333333')
            inset_axes_ax.yaxis.label.set_color('#333333')

            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(True)
            inset_axes_ax.yaxis.set_major_formatter(formatter)

            inset_axes_ax.errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="black", label='CNN-5', zorder=1)
            inset_axes_ax.errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3,       # ← caps horizontais de volta
                                      
                                      )  
            inset_axes_ax.legend(loc='upper left')
            x_inf_limite = -1
            x_sup_limite = 101
            y_inf_limite = 0
            y_sup_limite = 25
            inset_axes_ax.set_xlim([x_inf_limite, x_sup_limite])
            inset_axes_ax.set_ylim([y_inf_limite, y_sup_limite])

            #Baixo
            inset_axes_ax1 = inset_axes(
                ax[1], width="100%", height="100%",
                loc="upper right",
                bbox_to_anchor=(0.69, 0.17, 0.3, 0.8),
                bbox_transform=ax[1].transAxes
            )
            inset_axes_ax1.tick_params(axis='both', colors="#333333")
            inset_axes_ax1.xaxis.label.set_color('#333333')
            inset_axes_ax1.yaxis.label.set_color('#333333')

            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(True)
            inset_axes_ax1.yaxis.set_major_formatter(formatter)

            inset_axes_ax1.errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
            inset_axes_ax1.errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3, 
                                      
                                      )  
            inset_axes_ax1.legend(loc='upper left')
            x_inf_limite = -1
            x_sup_limite = 101
            y_inf_limite = 0
            y_sup_limite = 25
            inset_axes_ax1.set_xlim([x_inf_limite, x_sup_limite])
            inset_axes_ax1.set_ylim([y_inf_limite, y_sup_limite])
    
    ax[0].set_xlabel("Occupancy (%)", fontsize= fontSize)
    ax[0].set_ylabel(y_label, fontsize= fontSize)
    ax[0].errorbar(occupations_OF, means_OF, yerr=stds_OF, fmt='s', capsize=3, color=of_color, label='OF', zorder=1)
    ax[0].errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="black", label='CNN-5', zorder=1)
    _, caps8_0, bars8_0 = ax[0].errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3,       # ← caps horizontais de volta
                                      
                                      )    
    ax[0].legend(loc='upper left')
    ax[0].set_xlim(0,155)
    for bar in bars8_0:
        bar.set_linestyle('dashed')

    ax[1].errorbar(occupations_OF, means_OF, yerr=stds_OF, fmt='s', capsize=3, color=of_color, label='OF', zorder=1)
    ax[1].errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
    _, caps8_0, bars8_0 = ax[1].errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3       # ← caps horizontais de volta

                                      )    
    for bar in bars8_0:
        bar.set_linestyle('dashed')
    
    ax[1].set_xlabel("Occupancy (%)", fontsize= fontSize)
    ax[1].set_ylabel(y_label, fontsize= fontSize)
    ax[1].legend(loc='upper left')
    ax[1].set_xlim(0,155)

    plt.show()


#Plot_CNNxOF(metric='mean', zoom=True)
#Plot_CNNxOF(metric='std', zoom=True)

# Plot_CNN(metric="mean", zoom=True, box=True)
# Plot_CNN(metric="std", zoom=True, box=True)