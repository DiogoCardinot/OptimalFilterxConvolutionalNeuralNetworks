import numpy as np
import os
import matplotlib.pyplot as plt

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [50, 80]

def PlotErrors():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude", f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        if stats_por_intervalo_of:
            medias_of = [stats['media'] for stats in stats_por_intervalo_of.values()]
            labels_of = list(stats_por_intervalo_of.keys())

            medias_cnn = [stats['media'] for stats in stats_por_intervalo_cnn.values()]
            labels_cnn = list(stats_por_intervalo_cnn.keys())

            
            ax1.plot(range(len(labels_of)), medias_of, label='OF', marker='o', color='purple')
            ax1.plot(range(len(labels_cnn)), medias_cnn, label='CNN', marker='o', color='black')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Phase(ns) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Error Values\n(ns)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def PlotDispersion():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        stats_por_intervalo_of = data_of['stats_por_intervalo'].item()
        # # CNN
        data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude", f'errorxreal_{ocupacao}.npz')
        data_cnn = np.load(data_cnn_path, allow_pickle=True)
        stats_por_intervalo_cnn = data_cnn['stats_por_intervalo'].item()
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
        if stats_por_intervalo_of:
            desvios_of = [stats['std'] for stats in stats_por_intervalo_of.values()]
            labels_of = list(stats_por_intervalo_of.keys())

            desvios_cnn = [stats['std'] for stats in stats_por_intervalo_cnn.values()]
            labels_cnn = list(stats_por_intervalo_cnn.keys())

            ax1.plot(range(len(labels_of)), desvios_of, marker='o',  label='OF', color='purple')
            ax1.plot(range(len(labels_cnn)), desvios_cnn, marker='o',  label='CNN', color='black')
            ax1.set_xticks(range(len(labels_of)))
            ax1.set_xticklabels(labels_of, rotation=45, ha='right')
            ax1.set_xlabel(f'Real Phase(ns) - Occupancy {ocupacao}%')
            ax1.set_ylabel('Mean Dispersion Values\n(ns)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def PlotBoxPlots():
    for ocupacao in ocupacoes:
        # OF
        data_of_path = os.path.join(path, 'Dados', "OF_ErrorxRealPhase", f'errorxreal_{ocupacao}.npz')
        data_of = np.load(data_of_path, allow_pickle=True)
        erros_por_intervalo_of = data_of['erros_por_intervalo'].item()
        # CNN
        # data_cnn_path = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude", f'errorxreal_{ocupacao}.npz')
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


PlotErrors()
PlotDispersion()
PlotBoxPlots()
