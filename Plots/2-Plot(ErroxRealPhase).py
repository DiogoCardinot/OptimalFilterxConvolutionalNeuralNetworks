import numpy as np
import os
import matplotlib.pyplot as plt

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

n_janelamento = 7
ocupacoes = [50, 80]
CNN= 3

def PlotErrors():
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
        #     # ax1.plot(range(len(labels_real_amplitude)), medias_real_amplitude, label='Real Amplitude', marker='o', color='blue')
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
            ax1.plot(range(len(labels_real_amplitude)), medias_real_amplitude, label='Real Amplitude', marker='o', color='blue')
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
            ax1.plot(range(len(labels_real_amplitude)), desvios_real_amplitude, marker='o',  label='Real Amplitude', color='blue')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Amplitude(ADC Count) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Dispersion Values\n(ns)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def PlotBoxPlots():
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
        ax1.hist(of_error, bins = 50, alpha=0.7,histtype='step', color='purple', label='OF', linewidth=1)
        ax1.hist(cnn_error, bins = 50, alpha=0.7,histtype='step', color='black', label=f'CNN {CNN}', linewidth=1)
        ax1.hist(real_amplitude_error, bins = 50, alpha=0.7,histtype='step', color='blue', label='Real Amplitude', linewidth=1)

        ax1.set_xlabel(f'Phase estimation error (ns) - Occupancy {ocupacao}%')
        ax1.set_ylabel('Number of Events')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# PlotErrors()
# PlotDispersion()
# PlotBoxPlots()
PlotHistrograma()
