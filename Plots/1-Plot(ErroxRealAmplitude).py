import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [50, 80]
n_janelamento = 7

CNN=8

plt.rcParams['savefig.directory'] = os.path.dirname(path)

def PlotErrors():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealAmplitude", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude", f'janelamento_{n_janelamento}',f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        if stats_por_intervalo_of and stats_por_intervalo_cnn:
            medias_of = [stats['media'] for stats in stats_por_intervalo_of.values()]
            labels_of = list(stats_por_intervalo_of.keys())

            medias_cnn = [stats['media'] for stats in stats_por_intervalo_cnn.values()]
            labels_cnn = list(stats_por_intervalo_cnn.keys())
            
            ax1.plot(range(len(labels_of)), medias_of, marker='o', label='OF', color='purple')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.plot(range(len(labels_cnn)), medias_cnn, marker='o', label=f'CNN {CNN}', color='black')
            ax1.set_xticks(range(len(labels_cnn)))
            ax1.set_xticklabels(labels_cnn, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Amplitude (ADC counts) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Error Values\n(ADC counts)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def PlotDispersion():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealAmplitude",f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude",f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        if stats_por_intervalo_of and stats_por_intervalo_cnn:
            desvios_of = [stats['std'] for stats in stats_por_intervalo_of.values()]
            labels_of = list(stats_por_intervalo_of.keys())

            desvios_cnn = [stats['std'] for stats in stats_por_intervalo_cnn.values()]
            labels_cnn = list(stats_por_intervalo_cnn.keys())
            
            ax1.plot(range(len(labels_of)), desvios_of, marker='o', label='OF', color='purple')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.plot(range(len(labels_cnn)), desvios_cnn, marker='o', label=f'CNN {CNN}', color='black')
            ax1.set_xticks(range(len(labels_cnn)))
            ax1.set_xticklabels(labels_cnn, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Amplitude (ADC counts) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Dispersion Values\n(ADC counts)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def PlotBoxPlots():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealAmplitude", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        erros_por_intervalo_of = data_of['erros_por_intervalo'].item()
        # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude",f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        erros_por_intervalo_cnn = data_cnn['erros_por_intervalo'].item()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
        intervalos_of = list(erros_por_intervalo_of.keys())
        dados_boxplot_of = [erros_por_intervalo_of[intervalo] for intervalo in intervalos_of 
                        if erros_por_intervalo_of[intervalo]]
        intervalos_of_validos = [intervalo for intervalo in intervalos_of 
                                if erros_por_intervalo_of[intervalo]]
        

        intervalos_cnn = list(erros_por_intervalo_cnn.keys())
        dados_boxplot_cnn = [erros_por_intervalo_cnn[intervalo] for intervalo in intervalos_cnn 
                        if erros_por_intervalo_cnn[intervalo]]
        intervalos_cnn_validos = [intervalo for intervalo in intervalos_cnn 
                                if erros_por_intervalo_cnn[intervalo]]
        
        if dados_boxplot_of and dados_boxplot_cnn:
            bp = ax1.boxplot(dados_boxplot_of, tick_labels=intervalos_of_validos, medianprops=dict(color='purple', linewidth=1), label='OF')
            ax1.set_xticklabels(intervalos_of_validos, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Amplitude(ADC counts) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Error Values\n(ADC counts)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')

            bp_cnn = ax2.boxplot(dados_boxplot_cnn, tick_labels=intervalos_cnn_validos, medianprops=dict(color='black', linewidth=1), label=f'CNN {CNN}')
            ax2.set_xticklabels(intervalos_cnn_validos, rotation=45, ha='right')
            ax2.set_xlabel(f'Real Amplitude(ADC counts) - Occupancy {ocupacao}%')
            ax2.set_ylabel('Error Values\n(ADC counts)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')

        plt.tight_layout()
        plt.show()

def PlotHistrogramas():
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    of_color = '#9900ff'

    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # OF
        of_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'AmplitudeEstimada_OF', f'janelamento_{n_janelamento}',f'results_occupation_{ocupacao}.npz')      
        of_data = np.load(of_data_path)
        of_error = of_data['error']
        #CNN
        cnn_data_path = os.path.join(dataset_path,f'RedeNeuralConvolucional', f'CNN_{CNN}',f'results_ocupacao_{ocupacao}.npz')      
        cnn_data = np.load(cnn_data_path)
        cnn_error = cnn_data['error']
        
        bins = 150
        ax[idx].hist(of_error, bins = bins, alpha=0.7,histtype='step', color=of_color, linewidth=2)
        ax[idx].hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color=cnn_color, linewidth=2)
        ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+4, va='top')
        ax[idx].set_xlabel(f'Erro de estimação de amplitude (ADC Counts)', fontsize=fontSize)
        ax[idx].set_ylabel('Número de eventos', fontsize=fontSize)
        # ax[idx].legend(loc='best')
        ax[idx].grid(True, alpha=0.3)
        ax[idx].tick_params(axis='both', which='major', labelsize=14)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        ax[idx].yaxis.set_major_formatter(formatter)
        if idx==0:
            ax[idx].set_xlim(-75,50)
        elif idx==1:
            ax[idx].set_xlim(-200,100)
        elif idx==2:
            ax[idx].set_xlim(-300,100)
        elif idx==3:
            ax[idx].set_xlim(-300,200)

    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=of_color, linewidth=2))
    labels.append('OF')
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

# PlotErrors()
# PlotDispersion()
# PlotBoxPlots()
PlotHistrogramas()
