import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

n_janelamento = 7


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

def Plot_CNNxOF(metric):
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

    ax[0].errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="#1A1A1A", label='CNN-5', zorder=1)
    ax[0].errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
    # for i, (xi, yi, err) in enumerate(zip(x, y, yerr)):
    #     ax[0].plot([xi, xi], [yi - err, yi + err], linestyle='None', color=of_color, linewidth=2)
    #     ax[0].plot([xi - 0.2, xi + 0.2], [yi - err, yi - err], color=of_color, linewidth=2)
    #     ax[0].plot([xi - 0.2, xi + 0.2], [yi + err, yi + err], color=of_color, linewidth=2)
    
    if metric=='mean':
        y_label = 'Mean values\n(ADC counts)'
    elif metric=='std':
        y_label = 'Mean dispersion values\n(ADC counts)'
        
    ax[0].set_xlabel("Occupancy (%)", fontsize= fontSize-8)
    ax[0].set_ylabel(y_label, fontsize= fontSize-12)
    ax[0].legend(loc='best')
    ax[0].legend(loc='best')

    ax[1].errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
    ax[1].errorbar(x,y, yerr=yerr, fmt='s', capsize=3, color=of_color, label='OF', zorder=10)
    ax[1].set_xlabel("Occupancy (%)", fontsize= fontSize-12)
    ax[1].set_ylabel(y_label, fontsize= fontSize-12)
    ax[1].legend(loc='best')

    plt.show()

def Plot_CNN(metric):
    CNN_Data = SaveDataMeanSTD_CNN(metric)

    occupations_CNN3, means_CNN3, stds_CNN3 = GetMeanCNN(CNN_Data, 3)
    occupations_CNN5, means_CNN5, stds_CNN5 = GetMeanCNN(CNN_Data, 5)
    occupations_CNN8, means_CNN8, stds_CNN8 = GetMeanCNN(CNN_Data, 8)
    total_inches_image = 6.32
    fontSize = 24
    cnn8_color ="#006130"
    fig, ax = plt.subplots(2, 1, figsize=(total_inches_image, 4), constrained_layout=True)
    ax = ax.flatten()

    x = occupations_CNN8
    y = means_CNN8
    yerr = stds_CNN8

    ax[0].errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="#1A1A1A", label='CNN-5', zorder=1)
    _, caps8_0, bars8_0 = ax[0].errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3,       # ← caps horizontais de volta
                                      
                                      )    
    for bar in bars8_0:
        bar.set_linestyle('dashed')


    if metric=='mean':
        y_label = 'Mean values\n(ADC counts)'
    elif metric=='std':
        y_label = 'Mean dispersion values\n(ADC counts)'
        
    ax[0].set_xlabel("Occupancy (%)", fontsize= fontSize-8)
    ax[0].set_ylabel(y_label, fontsize= fontSize-12)
    ax[0].legend(loc='best')
    ax[0].legend(loc='best')

    ax[1].errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
    _, caps8_0, bars8_0 = ax[1].errorbar(x, y, yerr=yerr,
                                      fmt='*', color=cnn8_color, label='CNN-8',
                                      zorder=10,
                                      capsize=3       # ← caps horizontais de volta

                                      )    
    for bar in bars8_0:
        bar.set_linestyle('dashed')
    
    ax[1].set_xlabel("Occupancy (%)", fontsize= fontSize-12)
    ax[1].set_ylabel(y_label, fontsize= fontSize-12)
    ax[1].legend(loc='best')

    plt.show()


# Plot_CNNxOF(metric='mean')
# Plot_CNNxOF(metric='std')

Plot_CNN(metric="mean")
Plot_CNN(metric="std")