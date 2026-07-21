import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

n_janelamento = 7
ocupacoes = [0]
CNN= 3

plt.rcParams['savefig.directory'] = os.path.dirname(path)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def PlotError():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        stats_por_intervalo_of_amplitude = data_of['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealPhase",f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()

        # Real Amplitude
        data_real_amplitude_path = os.path.join(path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        # if stats_por_intervalo_of and stats_por_intervalo_cnn and stats_por_intervalo_real_amplitude:
        #     medias_of = [stats['media'] for stats in stats_por_intervalo_of.values()]
        #     labels_of = list(stats_por_intervalo_of.keys())

        #     medias_cnn = [stats['media'] for stats in stats_por_intervalo_cnn.values()]
        #     labels_cnn = list(stats_por_intervalo_cnn.keys())

        #     medias_real_amplitude = [stats['media'] for stats in stats_por_intervalo_real_amplitude.values()]
        #     labels_real_amplitude = list(stats_por_intervalo_real_amplitude.keys())

        #     ax1.plot(range(len(labels_of)), medias_of, label='OF', marker='o', color='purple')
        #     ax1.plot(range(len(labels_cnn)), medias_cnn, label=f'CNN {CNN}', marker='o', color='black')
        #     # ax1.plot(range(len(labels_real_amplitude)), medias_real_amplitude, label='Real Amplitude', marker='o', color='deepskyblue')
        #     ax1.set_xticks(range(len(labels_of)))
        #     ax1.set_xticklabels(labels_of, rotation=45, ha='right')
        #     ax1.set_xlabel(f'Real Phase(ns) - Occupancy {ocupacao}%')
        #     ax1.set_ylabel('Mean Error Values\n(ns)')
        #     ax1.legend(loc='best')
        #     ax1.grid(True, alpha=0.3)
        if stats_por_intervalo_of_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            medias_of = [stats['media'] for stats in stats_por_intervalo_of_amplitude.values()]
            labels_of = list(stats_por_intervalo_of_amplitude.keys())

            medias_cnn = [stats['media'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            medias_real_amplitude = [stats['media'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            ax1.plot(range(len(labels_of)), medias_of, label='OF', marker='o', color='purple')
            ax1.plot(range(len(labels_cnn)), medias_cnn, label=f'CNN {CNN}', marker='o', color='black')
            ax1.plot(range(len(labels_real_amplitude)), medias_real_amplitude, label='Real Amplitude', marker='o', color='deepskyblue')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Amplitude(ADC Count) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Error Values\n(ns)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def PlotDispersion():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        stats_por_intervalo_of_amplitude = data_of['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()

        # Real Amplitude
        data_real_amplitude_path = os.path.join(path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        if stats_por_intervalo_of_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            desvios_of = [stats['std'] for stats in stats_por_intervalo_of_amplitude.values()]
            labels_of = list(stats_por_intervalo_of_amplitude.keys())

            desvios_cnn = [stats['std'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            desvios_real_amplitude = [stats['std'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            ax1.plot(range(len(labels_of)), desvios_of, marker='o',  label='OF', color='purple')
            ax1.plot(range(len(labels_cnn)), desvios_cnn, marker='o',  label=f'CNN {CNN}', color='black')
            ax1.plot(range(len(labels_real_amplitude)), desvios_real_amplitude, marker='o',  label='Real Amplitude', color='deepskyblue')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Amplitude(ADC Count) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Dispersion Values\n(ns)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def PlotBoxPlot():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        erros_por_intervalo_of = data_of['erros_por_intervalo'].item()
        # CNN
        # data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude", f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        # data_cnn = np.load(data_cnn_path, allow_pickle=True)
        # erros_por_intervalo_cnn = data_cnn['erros_por_intervalo'].item()

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
            
        intervalos_of = list(erros_por_intervalo_of.keys())
        dados_boxplot_of = [erros_por_intervalo_of[intervalo] for intervalo in intervalos_of 
                        if erros_por_intervalo_of[intervalo]]
        intervalos_of_validos = [intervalo for intervalo in intervalos_of 
                                if erros_por_intervalo_of[intervalo]]
        
        if dados_boxplot_of:
            bp = ax1.boxplot(dados_boxplot_of, tick_labels=intervalos_of_validos, medianprops=dict(color='purple', linewidth=1), label='OF')
            ax1.set_xticklabels(intervalos_of_validos, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Phase(ns) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Error Values\n(ADC counts)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')

        plt.tight_layout()
        plt.show()

def PlotHistrograma():
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

    for ocupacao in ocupacoes:
        # OF
        of_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_OF', f'janelamento_{n_janelamento}',f'phase_of_occupation_{ocupacao}.npz')      
        of_data = np.load(of_data_path)
        of_error = of_data['error']
        #CNN
        cnn_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_{CNN}',f'phase_cnn_occupation_{ocupacao}.npz')      
        cnn_data = np.load(cnn_data_path)
        cnn_error = cnn_data['error']
        # Real Amplitude
        real_amplitude_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_RealAmplitude', f'janelamento_{n_janelamento}',f'phase_real_amplitude_occupation_{ocupacao}.npz')      
        real_amplitude_data = np.load(real_amplitude_data_path)
        real_amplitude_error = real_amplitude_data['error']

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        bins = 150
        ax1.hist(of_error, bins = bins, alpha=0.7,histtype='step', color='purple', label='OF', linewidth=1)
        ax1.hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color='black', label=f'CNN {CNN}', linewidth=1)
        ax1.hist(real_amplitude_error, bins = bins, alpha=0.7,histtype='step', color='deepskyblue', label='Real Amplitude', linewidth=1)

        ax1.set_xlabel(f'Phase estimation error (ns) - Occupancy {ocupacao}%')
        ax1.set_ylabel('Number of Events')
        ax1.legend(loc='best')
        ax1.set_xlim(-600,600)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def PlotHistrogramas(zoom):
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"
    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
            cnn_estimated_color = "deepskyblue"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"
            cnn_estimated_color = "darkorange"

        # OF
        of_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_OF', f'janelamento_{n_janelamento}',f'phase_of_occupation_{ocupacao}.npz')      
        of_data = np.load(of_data_path)
        of_error = of_data['error']
        #CNN
        cnn_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_{CNN}',f'phase_cnn_occupation_{ocupacao}.npz')      
        cnn_data = np.load(cnn_data_path)
        cnn_error = cnn_data['error']
        # CNN Fase Estimada
        cnn_data_path_estimated = os.path.join(dataset_path, f'RedeNeuralConvolucional_Fase', f'CNN_{CNN}', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')      
        cnn_data_estimated = np.load(cnn_data_path_estimated)
        cnn_error_estimated = cnn_data_estimated['error']
        # Real Amplitude
        real_amplitude_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_RealAmplitude', f'janelamento_{n_janelamento}',f'phase_real_amplitude_occupation_{ocupacao}.npz')      
        real_amplitude_data = np.load(real_amplitude_data_path)
        real_amplitude_error = real_amplitude_data['error']

        bins = 150
        ax[idx].hist(of_error, bins = bins, alpha=0.7,histtype='step', color=of_color, linewidth=2)
        ax[idx].hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color=cnn_color, linewidth=3)
        # ax[idx].hist(cnn_error_estimated, bins = bins, alpha=0.7,histtype='step', color=cnn_estimated_color, linewidth=2)
        ax[idx].hist(real_amplitude_error, bins = bins, alpha=0.7,histtype='step', color=real_amplitude_color, linewidth=2, linestyle='dashed')
        ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+4, va='top')
        ax[idx].set_xlabel(f'Erro de estimação de fase (ns)', fontsize=fontSize-2)
        ax[idx].set_ylabel('Número de eventos', fontsize=fontSize-2)
        ax[idx].grid(True, alpha=0.3)
        ax[idx].tick_params(axis='both', which='major', labelsize=20)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        ax[idx].yaxis.set_major_formatter(formatter)
        ax[idx].set_xlim(-500,500)
        if zoom:
            # Adicionando janelas de zoom
            inset_axes_ax = inset_axes(
                ax[idx], width="100%", height="100%",
                loc="upper right",
                bbox_to_anchor=(0.69, 0.17, 0.3, 0.8),
                bbox_transform=ax[idx].transAxes
            )
            inset_axes_ax.tick_params(axis='both', colors="#333333")
            inset_axes_ax.xaxis.label.set_color('#333333')
            inset_axes_ax.yaxis.label.set_color('#333333')

            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(True)
            inset_axes_ax.yaxis.set_major_formatter(formatter)

            inset_axes_ax.hist(cnn_error_estimated, bins=bins, alpha=0.7, histtype='step', color=cnn_estimated_color, linewidth=2)

            if idx==0:
                x_inf_limite = -0.5
                x_sup_limite = 0.5
                y_inf_limite = 0
                y_sup_limite = 1.75*10**6
            elif idx==1:
                x_inf_limite = -5
                x_sup_limite = 5
                y_inf_limite = 0
                y_sup_limite = 7*10**5
            elif idx==2:
                x_inf_limite = -5
                x_sup_limite = 5
                y_inf_limite = 0
                y_sup_limite = 2.5*10**5
            elif idx==3:
                x_inf_limite = -6
                x_sup_limite = 6
                y_inf_limite = 0
                y_sup_limite = 2.5*10**4


            inset_axes_ax.set_xlim([x_inf_limite, x_sup_limite])
            inset_axes_ax.set_ylim([y_inf_limite, y_sup_limite])

    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=of_color, linewidth=2))
    labels.append('OF')
    handles.append(plt.Line2D([0], [0], color=real_amplitude_color, linewidth=2, linestyle='dashed'))
    labels.append('Real Amplitude')
    cnn_types = [r'CNN-3', r'CNN-5']
    unique_cnns = list(set(cnn_types))
    for cnn_type in unique_cnns:
        if cnn_type == "CNN-5":
            color = "#1A1A1A"
        else: 
            color = "#B0B0B0"
       
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(cnn_type)
    
    # cnn_estimated_types = ['CNN-3', 'CNN-5']
    # unique_estimated_cnns = list(set(cnn_estimated_types))
    # for cnn_type in unique_estimated_cnns:
    #     if cnn_type == "CNN-5":
    #         color = "deepskyblue"
    #     else: 
    #         color = "darkorange"
       
    #     handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
    #     labels.append(cnn_type)

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.49, 0.9999),
        frameon=False,
        fontsize=fontSize-1
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
    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"
    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        stats_por_intervalo_of_amplitude = data_of['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealPhase",f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()

        # Real Amplitude
        data_real_amplitude_path = os.path.join(path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()

        if stats_por_intervalo_of_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            medias_of = [stats['media'] for stats in stats_por_intervalo_of_amplitude.values()]
            labels_of = list(stats_por_intervalo_of_amplitude.keys())

            medias_cnn = [stats['media'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            medias_real_amplitude = [stats['media'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            ax[idx].plot(range(len(labels_of)), medias_of, marker='o', color=of_color)
            ax[idx].plot(range(len(labels_cnn)), medias_cnn, marker='o', color=cnn_color)
            ax[idx].plot(range(len(labels_real_amplitude)), medias_real_amplitude, linestyle=':', marker='*', color=real_amplitude_color)
            ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+6, va='top')
            ax[idx].set_xticks(range(len(labels_of)))
            ax[idx].set_xticklabels(labels_of, rotation=45, ha='right',fontsize=fontSize-17)
            ax[idx].set_xlabel(f'Real Amplitude(ADC Counts)', fontsize=fontSize-8)
            ax[idx].set_ylabel('Mean Error Values\nPhase Estimation (ns)', fontsize=fontSize-8)
            ax[idx].grid(True, alpha=0.3)
    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=of_color, linewidth=2))
    labels.append('OF')
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
    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"

    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        stats_por_intervalo_of_amplitude = data_of['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()

        # Real Amplitude
        data_real_amplitude_path = os.path.join(path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()
        
        if stats_por_intervalo_of_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            desvios_of = [stats['std'] for stats in stats_por_intervalo_of_amplitude.values()]
            labels_of = list(stats_por_intervalo_of_amplitude.keys())

            desvios_cnn = [stats['std'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            desvios_real_amplitude = [stats['std'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            ax[idx].plot(range(len(labels_of)), desvios_of, marker='o',   color=of_color)
            ax[idx].plot(range(len(labels_cnn)), desvios_cnn, marker='o',   color=cnn_color)
            ax[idx].plot(range(len(labels_real_amplitude)), desvios_real_amplitude, linestyle=':', marker='*', color=real_amplitude_color)
            ax[idx].set_xticks(range(len(labels_of)))
            ax[idx].set_xticklabels(labels_of, rotation=45, ha='right',fontsize=fontSize-17)
            ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+4, va='top')
            ax[idx].tick_params(axis='y', which='major', labelsize=14)

            ax[idx].set_xlabel(f'Real Amplitude (ADC Counts)', fontsize = fontSize-8)
            ax[idx].set_ylabel('Mean Dispersion Values\nPhase Estimation (ns)', fontsize = fontSize-8)
            ax[idx].grid(True, alpha=0.3)
    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=of_color, linewidth=2))
    labels.append('OF')
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

def filtrar_ate_750(labels, desvios, limite=750):
    indices = [i for i, label in enumerate(labels) if float(label.split(' \u2013 ')[1]) <= limite]
    return [labels[i] for i in indices], [desvios[i] for i in indices]


def PlotDispersions1():
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"

    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"

        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        stats_por_intervalo_of_amplitude = data_of['stats_por_intervalo_amplitude'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_{CNN}', f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        stats_por_intervalo_cnn_amplitude = data_cnn['stats_por_intervalo_amplitude'].item()
        # CNN Estimada
        
        # Real Amplitude
        data_real_amplitude_path = os.path.join(path, "Dados", "RealAmplitude_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'errorxreal_{ocupacao}.npz')
        data_real_amplitude = np.load(data_real_amplitude_path, allow_pickle=True)
        stats_por_intervalo_real_amplitude = data_real_amplitude['stats_por_intervalo'].item()
        stats_por_intervalo_real_amplitude_ = data_real_amplitude['stats_por_intervalo_amplitude'].item()
        
        if stats_por_intervalo_of_amplitude and stats_por_intervalo_cnn_amplitude and stats_por_intervalo_real_amplitude_:
            desvios_of = [stats['std'] for stats in stats_por_intervalo_of_amplitude.values()]
            labels_of = list(stats_por_intervalo_of_amplitude.keys())

            desvios_cnn = [stats['std'] for stats in stats_por_intervalo_cnn_amplitude.values()]
            labels_cnn = list(stats_por_intervalo_cnn_amplitude.keys())

            desvios_real_amplitude = [stats['std'] for stats in stats_por_intervalo_real_amplitude_.values()]
            labels_real_amplitude = list(stats_por_intervalo_real_amplitude_.keys())

            labels_of, desvios_of = filtrar_ate_750(labels_of, desvios_of)
            labels_cnn, desvios_cnn = filtrar_ate_750(labels_cnn, desvios_cnn)
            labels_real_amplitude, desvios_real_amplitude = filtrar_ate_750(labels_real_amplitude, desvios_real_amplitude)

            ax[idx].plot(range(len(labels_of)), desvios_of, marker='o',   color=of_color)
            ax[idx].plot(range(len(labels_cnn)), desvios_cnn, marker='o',   color=cnn_color)
            ax[idx].plot(range(len(labels_real_amplitude)), desvios_real_amplitude, linestyle=':', marker='*', color=real_amplitude_color)
            ax[idx].set_xticks(range(len(labels_of)))
            ax[idx].set_xticklabels(labels_of, rotation=25, ha='right',fontsize=fontSize-12)
            ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+4, va='top')
            ax[idx].tick_params(axis='y', which='major', labelsize=14)

            ax[idx].set_xlabel(f'Real Amplitude (ADC Counts)', fontsize = fontSize-8)
            ax[idx].set_ylabel('Mean Dispersion Values\nPhase Estimation (ns)', fontsize = fontSize-8)
            ax[idx].grid(True, alpha=0.3)
    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=of_color, linewidth=2))
    labels.append('OF')
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
    plt.subplots_adjust(hspace=0.5, top=0.90)
    plt.show()

# PlotError()
# PlotDispersion()
# PlotHistrograma()
# PlotHistrogramas(zoom=False)
# PlotErros()
# PlotDispersions()
# PlotDispersions1()

def PlotHistrogramas1(zoom, box):
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")
    ocupacoes = [10,50,80,100]
    total_inches_image = 6.32
    fontSize = 24
    fig, (ax) = plt.subplots(2, 2, figsize=(15, 6))
    ax = ax.flatten()
    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"
    for idx, ocupacao in enumerate(ocupacoes):
        if ocupacao == 10 or ocupacao==50:
            CNN = 5
            cnn_color = "#1A1A1A"
            cnn_estimated_color = "deepskyblue"
        elif ocupacao==80 or ocupacao==100:
            CNN=3
            cnn_color = "#B0B0B0"
            cnn_estimated_color = "darkorange"

        # OF
        of_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_OF', f'janelamento_{n_janelamento}',f'phase_of_occupation_{ocupacao}.npz')      
        of_data = np.load(of_data_path)
        of_error = of_data['error']
        #CNN
        cnn_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_{CNN}',f'phase_cnn_occupation_{ocupacao}.npz')      
        cnn_data = np.load(cnn_data_path)
        cnn_error = cnn_data['error']
        # CNN Fase Estimada
        cnn_data_path_estimated = os.path.join(dataset_path, f'RedeNeuralConvolucional_Fase', f'CNN_{CNN}', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')      
        cnn_data_estimated = np.load(cnn_data_path_estimated)
        cnn_error_estimated = cnn_data_estimated['error']
        # Real Amplitude
        real_amplitude_data_path = os.path.join(dataset_path,f'FiltroOtimo',f'FaseEstimada_RealAmplitude', f'janelamento_{n_janelamento}',f'phase_real_amplitude_occupation_{ocupacao}.npz')      
        real_amplitude_data = np.load(real_amplitude_data_path)
        real_amplitude_error = real_amplitude_data['error']

        bins = 150
        ax[idx].hist(of_error, bins = bins, alpha=0.7,histtype='step', color=of_color, linewidth=2)
        ax[idx].hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color=cnn_color, linewidth=3)
        # ax[idx].hist(cnn_error_estimated, bins = bins, alpha=0.7,histtype='step', color=cnn_estimated_color, linewidth=2)
        ax[idx].hist(real_amplitude_error, bins = bins, alpha=0.7,histtype='step', color=real_amplitude_color, linewidth=2, linestyle='dashed')
        ax[idx].text(-0.15, 1.12, f'({chr(97+idx)})', transform=ax[idx].transAxes, fontsize=fontSize+4, va='top')
        ax[idx].set_xlabel(f'Erro de estimação de fase (ns)', fontsize=fontSize-2)
        ax[idx].set_ylabel('Número de eventos', fontsize=fontSize-2)
        ax[idx].grid(True, alpha=0.3)
        ax[idx].tick_params(axis='both', which='major', labelsize=20)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        ax[idx].yaxis.set_major_formatter(formatter)
        ax[idx].set_xlim(-500,500)
        if zoom:
            # Adicionando janelas de zoom
            inset_axes_ax = inset_axes(
                ax[idx], width="100%", height="100%",
                loc="upper right",
                bbox_to_anchor=(0.69, 0.17, 0.3, 0.8),
                bbox_transform=ax[idx].transAxes
            )
            inset_axes_ax.tick_params(axis='both', colors="#333333")
            inset_axes_ax.xaxis.label.set_color('#333333')
            inset_axes_ax.yaxis.label.set_color('#333333')

            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(True)
            inset_axes_ax.yaxis.set_major_formatter(formatter)

            inset_axes_ax.hist(cnn_error_estimated, bins=bins, alpha=0.7, histtype='step', color=cnn_estimated_color, linewidth=2)

            if idx==0:
                x_inf_limite = -0.5
                x_sup_limite = 0.5
                y_inf_limite = 0
                y_sup_limite = 1.75*10**6
            elif idx==1:
                x_inf_limite = -5
                x_sup_limite = 5
                y_inf_limite = 0
                y_sup_limite = 7*10**5
            elif idx==2:
                x_inf_limite = -5
                x_sup_limite = 5
                y_inf_limite = 0
                y_sup_limite = 2.5*10**5
            elif idx==3:
                x_inf_limite = -6
                x_sup_limite = 6
                y_inf_limite = 0
                y_sup_limite = 2.5*10**4


            inset_axes_ax.set_xlim([x_inf_limite, x_sup_limite])
            inset_axes_ax.set_ylim([y_inf_limite, y_sup_limite])
        
        if box and idx!=3:
            x_0 = 0.6  #posicao inicial em x
            y_0 = 0.6 #posicao inicial em y
            width = 0.3
            heigth = 0.3
            if idx ==0:
                x1,x2= -5, 16
                y1,y2 = 9.5*(10**5), 9.525*(10**5)
            elif idx==1:
                x1,x2= -4.5, 18
                y1,y2 = 5.49*10**5, 5.515*10**5
            elif idx==2:
                x1,x2= -22,22
                y1,y2 = 5.17*10**5, 5.5*10**5

            axins = inset_axes(ax[idx], width="100%", height="100%", bbox_to_anchor=(x_0, y_0, width, heigth),  # (x, y) posição do canto
                   bbox_transform=ax[idx].transAxes,
                   loc='center')
    
            axins.set_xticks([x1,x2])
            axins.set_yticks([y1,y2])
            axins.yaxis.tick_right()

            axins.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
            axins.tick_params(axis='both', colors="#424242")

            axins.hist(real_amplitude_error, bins = bins, alpha=0.7,histtype='step', color=real_amplitude_color, linewidth=2, linestyle='dashed')
            axins.hist(cnn_error, bins = bins, alpha=0.7,histtype='step', color=cnn_color, linewidth=3)
            axins.set_xlim(x1,x2)
            axins.set_ylim(y1,y2)
            plt.setp(axins.get_xticklabels(which='both'), fontsize=8)
            plt.setp(axins.get_yticklabels(), fontsize=8)
            mark_inset(ax[idx], axins, loc1=2, loc2=3, fc="none", ec="gray", linewidth=1.5)

    handles = []
    labels = []
    
    handles.append(plt.Line2D([0], [0], color=of_color, linewidth=2))
    labels.append('OF')
    handles.append(plt.Line2D([0], [0], color=real_amplitude_color, linewidth=2, linestyle='dashed'))
    labels.append('Real Amplitude')
    cnn_types = [r'CNN-3', r'CNN-5']
    unique_cnns = list(set(cnn_types))
    for cnn_type in unique_cnns:
        if cnn_type == "CNN-5":
            color = "#1A1A1A"
        else: 
            color = "#B0B0B0"
       
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(cnn_type)
    
    # cnn_estimated_types = ['CNN-3', 'CNN-5']
    # unique_estimated_cnns = list(set(cnn_estimated_types))
    # for cnn_type in unique_estimated_cnns:
    #     if cnn_type == "CNN-5":
    #         color = "deepskyblue"
    #     else: 
    #         color = "darkorange"
       
    #     handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
    #     labels.append(cnn_type)

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.49, 0.9999),
        frameon=False,
        fontsize=fontSize-1
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

PlotHistrogramas1(zoom= False, box= True)