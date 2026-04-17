import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

n_janelamento = 7
ocupacoes = [0]
CNN= 3

plt.rcParams['savefig.directory'] = os.path.dirname(path)

def PlotHistrogramasAmpitude():
    base_path = os.path.dirname(os.path.dirname(path))
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    cnn8_color ="#006130"

    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # CNN8
        cnn8_data_path = os.path.join(base_path,f'RedeNeuralConvolucional', f'CNN_8',f'results_ocupacao_{ocupacao}.npz')      
        cnn8_data = np.load(cnn8_data_path)
        cnn8_error = cnn8_data['error']
        #CNN
        cnn_data_path = os.path.join(base_path,f'RedeNeuralConvolucional', f'CNN_{CNN}',f'results_ocupacao_{ocupacao}.npz')      
        cnn_data = np.load(cnn_data_path)
        cnn_error = cnn_data['error']
        
        bins = 150
        ax[idx].hist(cnn8_error, bins = bins, alpha=0.7,histtype='step', color=cnn8_color, linewidth=2, linestyle='dotted')
        ax[idx].hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color=cnn_color, linewidth=2)
        ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+6, va='top')
        ax[idx].set_xlabel(f'Amplitude estimation error (ADC Counts)', fontsize=fontSize)
        ax[idx].set_ylabel('Number of Events', fontsize=fontSize)
        # ax[idx].legend(loc='best')
        ax[idx].grid(True, alpha=0.3)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        ax[idx].yaxis.set_major_formatter(formatter)
        if idx==0:
            ax[idx].set_xlim(-20,20)
        elif idx==1:
            ax[idx].set_xlim(-75,75)
        else:
            ax[idx].set_xlim(-100,100)
        # elif idx==1:
        #     ax[idx].set_xlim(-200,100)
        # elif idx==2:
        #     ax[idx].set_xlim(-300,100)
        # elif idx==3:
        #     ax[idx].set_xlim(-300,200)

    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=cnn8_color, linewidth=2, linestyle='dotted'))
    labels.append('CNN-8')
    cnn_types = ['CNN-3', 'CNN-5']
    unique_cnns = list(set(cnn_types))
    for cnn_type in unique_cnns:
        if cnn_type == "CNN-5":
            color = "#1A1A1A"
        else: 
            color = "#B0B0B0"
       
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(cnn_type)

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.9999),
        frameon=False,
        fontsize=fontSize
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def PlotHistrogramasPhase():
    base_path = os.path.dirname(os.path.dirname(path))
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    cnn8_color ="#006130"
    real_amplitude_color = "#FA3232"
    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # CNN8
        cnn8_data_path = os.path.join(base_path,f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_8',f'phase_cnn_occupation_{ocupacao}.npz')      
        cnn8_data = np.load(cnn8_data_path)
        cnn8_error = cnn8_data['error']
        #CNN
        cnn_data_path = os.path.join(base_path,f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_{CNN}',f'phase_cnn_occupation_{ocupacao}.npz')      
        cnn_data = np.load(cnn_data_path)
        cnn_error = cnn_data['error']
        # Real Amplitude
        real_amplitude_data_path = os.path.join(base_path,f'FiltroOtimo',f'FaseEstimada_RealAmplitude', f'janelamento_{n_janelamento}',f'phase_real_amplitude_occupation_{ocupacao}.npz')      
        real_amplitude_data = np.load(real_amplitude_data_path)
        real_amplitude_error = real_amplitude_data['error']

        bins = 150
        ax[idx].hist(cnn8_error, bins = bins, alpha=0.7,histtype='step', color=cnn8_color, linewidth=2, linestyle='dotted')
        ax[idx].hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color=cnn_color, linewidth=2)
        ax[idx].hist(real_amplitude_error, bins = bins, alpha=0.7,histtype='step', color=real_amplitude_color, linewidth=2, linestyle='dashed')
        ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+6, va='top')
        ax[idx].set_xlabel(f'Phase estimation error (ns)', fontsize=fontSize)
        ax[idx].set_ylabel('Number of Events', fontsize=fontSize)
        ax[idx].grid(True, alpha=0.3)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        ax[idx].yaxis.set_major_formatter(formatter)
        ax[idx].set_xlim(-400,400)
    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=cnn8_color, linewidth=2, linestyle='dotted'))
    labels.append('CNN-8')
    handles.append(plt.Line2D([0], [0], color=real_amplitude_color, linewidth=2, linestyle='dashed'))
    labels.append('Real Amplitude')
    cnn_types = ['CNN-3', 'CNN-5']
    unique_cnns = list(set(cnn_types))
    for cnn_type in unique_cnns:
        if cnn_type == "CNN-5":
            color = "#1A1A1A"
        else: 
            color = "#B0B0B0"
       
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(cnn_type)

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.9999),
        frameon=False,
        fontsize=fontSize
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def PlotErros():
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    cnn8_color = "#006130"
    real_amplitude_color = "#FA3232"
    base_path = os.path.dirname(os.path.dirname(path))
    plots_path = os.path.join(base_path, "Plots")
    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # CNN8
        data_cnn8_path = os.path.join(plots_path, 'Dados', 'CNN_ErrorxRealPhase', f'janelamento_{n_janelamento}', 'CNN_8', f'errorxreal_{ocupacao}.npz')
        data_cnn8 = np.load(data_cnn8_path, allow_pickle=True)
        stats_por_intervalo_cnn8 = data_cnn8['stats_por_intervalo'].item()
        stats_por_intervalo_cnn8_amplitude = data_cnn8['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(plots_path, "Dados", "CNN_ErrorxRealPhase",f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()

        # Real Amplitude
        data_real_amplitude_path = os.path.join(plots_path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()

        if stats_por_intervalo_cnn8_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            medias_cnn8 = [stats['media'] for stats in stats_por_intervalo_cnn8_amplitude.values()]
            labels_cnn8 = list(stats_por_intervalo_cnn8_amplitude.keys())

            medias_cnn = [stats['media'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            medias_real_amplitude = [stats['media'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            ax[idx].plot(range(len(labels_cnn8)), medias_cnn8, marker='o', color=cnn8_color)
            ax[idx].plot(range(len(labels_cnn)), medias_cnn, marker='o', color=cnn_color)
            ax[idx].plot(range(len(labels_real_amplitude)), medias_real_amplitude, linestyle=':', marker='*', color=real_amplitude_color)
            ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+6, va='top')
            ax[idx].set_xticks(range(len(labels_cnn8)))
            ax[idx].set_xticklabels(labels_cnn8, rotation=45, ha='right',fontsize=fontSize-17)
            ax[idx].set_xlabel(f'Real Amplitude(ADC Counts)', fontsize=fontSize-8)
            ax[idx].set_ylabel('Mean Error Values\nPhase Estimation (ns)', fontsize=fontSize-8)
            ax[idx].grid(True, alpha=0.3)
    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=cnn8_color, linewidth=2))
    labels.append('CNN-8')
    handles.append(plt.Line2D([0], [0], color=real_amplitude_color, linewidth=2, linestyle=':', marker='*',))
    labels.append('Real Amplitude')
    cnn_types = ['CNN-3', 'CNN-5']
    unique_cnns = list(set(cnn_types))
    for cnn_type in unique_cnns:
        if cnn_type == "CNN-5":
            color = "#1A1A1A"
        else: 
            color = "#B0B0B0"
       
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(cnn_type)

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.9999),
        frameon=False,
        fontsize=fontSize
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def PlotDispersions():
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    cnn8_color = "#006130"
    real_amplitude_color = "#FA3232"
    base_path = os.path.dirname(os.path.dirname(path))
    plots_path = os.path.join(base_path, "Plots")

    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # CNN8
        data_cnn8_path = os.path.join(plots_path, "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_8', f'errorxreal_{ocupacao}.npz')
        data_cnn8 = np.load(data_cnn8_path, allow_pickle=True)
        stats_por_intervalo_cnn8 = data_cnn8['stats_por_intervalo'].item()
        stats_por_intervalo_cnn8_amplitude = data_cnn8['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(plots_path, "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()

        # Real Amplitude
        data_real_amplitude_path = os.path.join(plots_path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()
        
        if stats_por_intervalo_cnn8_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            desvios_cnn8 = [stats['std'] for stats in stats_por_intervalo_cnn8_amplitude.values()]
            labels_cnn8 = list(stats_por_intervalo_cnn8_amplitude.keys())

            desvios_cnn = [stats['std'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            desvios_real_amplitude = [stats['std'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            ax[idx].plot(range(len(labels_cnn8)), desvios_cnn8, marker='o',   color=cnn8_color)
            ax[idx].plot(range(len(labels_cnn)), desvios_cnn, marker='o',   color=cnn_color)
            ax[idx].plot(range(len(labels_real_amplitude)), desvios_real_amplitude, linestyle=':', marker='*', color=real_amplitude_color)
            ax[idx].set_xticks(range(len(labels_cnn8)))
            ax[idx].set_xticklabels(labels_cnn8, rotation=45, ha='right',fontsize=fontSize-17)
            ax[idx].tick_params(axis='y', which='major', labelsize=14)
            ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+6, va='top')
            ax[idx].set_xlabel(f'Real Amplitude(ADC Counts)', fontsize = fontSize-8)
            ax[idx].set_ylabel('Mean Dispersion Values\nPhase Estimation (ns)', fontsize = fontSize-8)
            ax[idx].grid(True, alpha=0.3)
    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=cnn8_color, linewidth=2))
    labels.append('CNN-8')
    handles.append(plt.Line2D([0], [0], color=real_amplitude_color, linewidth=2, linestyle=':', marker='*',))
    labels.append('Real Amplitude')
    cnn_types = ['CNN-3', 'CNN-5']
    unique_cnns = list(set(cnn_types))
    for cnn_type in unique_cnns:
        if cnn_type == "CNN-5":
            color = "#1A1A1A"
        else: 
            color = "#B0B0B0"
       
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(cnn_type)

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.9999),
        frameon=False,
        fontsize=fontSize
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def SaveDataMeanSTD():
    occupation_new = [0,10,20,30,40,50,60,70,80,90,100]
    CNN_paths = [3,5,8]
    base_path = os.path.dirname(os.path.dirname(path))
    
    data_cnn_path = os.path.join(base_path, "RedeNeuralConvolucional")
    CNN_Data = {}
    for cnn in CNN_paths:
        CNN_path =  os.path.join(data_cnn_path, f"CNN_{cnn}")
        for occupation in occupation_new:
            CNN_path_complete = os.path.join(CNN_path, f"results_ocupacao_{occupation}.npz")
            CNN_data = np.load(CNN_path_complete)
            #{(CNN architecture, occupation): (mean error, std error)}
            CNN_Data[(cnn, occupation)] = (CNN_data['mean_error'], CNN_data['std_error'])

    return CNN_Data


def GetMeanSTDCNN(CNN_Data, cnn_target):
    filtrado = {
        occ: (mean, std)
        for (cnn, occ), (mean, std) in CNN_Data.items()
        if cnn == cnn_target
    }

    occupations = list(filtrado.keys())
    means = [filtrado[o][0] for o in occupations]
    stds  = [filtrado[o][1] for o in occupations]

    return occupations, means, stds


def PlotMeanSTD():
    CNN_Data = SaveDataMeanSTD()

    occupations_CNN3, means_CNN3, stds_CNN3 = GetMeanSTDCNN(CNN_Data, 3)
    occupations_CNN5, means_CNN5, stds_CNN5 = GetMeanSTDCNN(CNN_Data, 5)
    occupations_CNN8, means_CNN8, stds_CNN8 = GetMeanSTDCNN(CNN_Data, 8)
    total_inches_image = 6.32
    fontSize = 24
    cnn8_color ="#006130"
    fig, ax = plt.subplots(2, 1, figsize=(total_inches_image, 4), constrained_layout=True)
    ax = ax.flatten()

    x = occupations_CNN8
    y = means_CNN8
    yerr = stds_CNN8

    ax[0].errorbar(occupations_CNN5, means_CNN5, yerr=stds_CNN5, fmt='s', capsize=3, color="#1A1A1A", label='CNN-5', zorder=1)
    ax[0].plot(x, y, linestyle='dashed', marker='*', color=cnn8_color, label='CNN-8', zorder=10)
    for i, (xi, yi, err) in enumerate(zip(x, y, yerr)):
        ax[0].plot([xi, xi], [yi - err, yi + err], linestyle='dotted', color=cnn8_color, linewidth=2)
        ax[0].plot([xi - 0.2, xi + 0.2], [yi - err, yi - err], color=cnn8_color, linewidth=2)
        ax[0].plot([xi - 0.2, xi + 0.2], [yi + err, yi + err], color=cnn8_color, linewidth=2)
    ax[0].set_xlabel("Occupancy (%)", fontsize= fontSize-8)
    ax[0].set_ylabel("Mean values\n(ADC counts)", fontsize= fontSize-8)
    ax[0].legend(loc='best')
    ax[0].set_xlabel("Occupancy (%)", fontsize= fontSize-8)
    ax[0].set_ylabel("Mean values\n(ADC counts)", fontsize= fontSize-8)
    ax[0].legend(loc='best')

    ax[1].errorbar(occupations_CNN3, means_CNN3, yerr=stds_CNN3, fmt='s', capsize=3, color='#B0B0B0', label='CNN-3', zorder=0)
    ax[1].plot(x, y, linestyle='dashed', marker='*', color=cnn8_color, label='CNN-8', zorder=10)
    for i, (xi, yi, err) in enumerate(zip(x, y, yerr)):
        ax[1].plot([xi, xi], [yi - err, yi + err], linestyle='dotted', color=cnn8_color, linewidth=2)
        ax[1].plot([xi - 0.2, xi + 0.2], [yi - err, yi - err], color=cnn8_color, linewidth=2)
        ax[1].plot([xi - 0.2, xi + 0.2], [yi + err, yi + err], color=cnn8_color, linewidth=2)
    ax[1].set_xlabel("Occupancy (%)", fontsize= fontSize-8)
    ax[1].set_ylabel("Mean values\n(ADC counts)", fontsize= fontSize-8)
    ax[1].legend(loc='best')

    plt.show()

# PlotHistrogramasAmpitude()
# PlotHistrogramasPhase()
# PlotErros()
PlotDispersions()
# PlotMeanSTD()