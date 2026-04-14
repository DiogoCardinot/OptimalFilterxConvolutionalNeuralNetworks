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

def PlotHistrogramas():
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

PlotHistrogramas()