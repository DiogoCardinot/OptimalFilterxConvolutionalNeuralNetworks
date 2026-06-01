import numpy as np
import os
import matplotlib.pyplot as plt

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7

base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

plt.rcParams['savefig.directory'] = os.path.dirname(path)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def PlotFaseDispersionOcupacao():
    of_dispersions = []
    cnn3_dispersions = []
    cnn5_dispersions = []
    real_amplitude_dispersions = []
    cnn3_estimated_dispersions = []
    cnn5_estimated_dispersions = []
    for ocupacao in ocupacoes:
        # PATHS
        # TEM QUE PEGAR DA PASTA DO FILTRO OTIMO..;
        of_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_OF', f'janelamento_{n_janelamento}', f'phase_of_occupation_{ocupacao}.npz')
        cnn3_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_3', f'phase_cnn_occupation_{ocupacao}.npz' )
        cnn5_path = os.path.join(dataset_path, "FiltroOtimo", f'FaseEstimada_CNN', f'janelamento_{n_janelamento}',f'CNN_5', f'phase_cnn_occupation_{ocupacao}.npz' )
        real_amplitude_path = os.path.join(dataset_path, "FiltroOtimo", f"FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}', f'phase_real_amplitude_occupation_{ocupacao}.npz' )
        cnn3_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_3', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')
        cnn5_estimated_path = os.path.join(dataset_path, "RedeNeuralConvolucional_Fase", f'CNN_5', f'janelamento_{n_janelamento}', f'results_ocupacao_{ocupacao}.npz')

        # LOADS
        of_load = np.load(of_path)
        cnn3_load = np.load(cnn3_path)
        cnn5_load = np.load(cnn5_path)
        real_amplitude_load = np.load(real_amplitude_path)
        cnn3_estimated_load = np.load(cnn3_estimated_path)
        cnn5_estimated_load = np.load(cnn5_estimated_path)

        # DATA LOAD
        data_of = of_load['std']
        data_cnn3 = cnn3_load['std']
        data_cnn5 = cnn5_load['std']
        data_real_amplitude = real_amplitude_load['std']
        data_cnn3_estimate = cnn3_estimated_load['std_error']
        data_cnn5_estimate = cnn5_estimated_load['std_error']

        of_dispersions.append(data_of)
        cnn3_dispersions.append(data_cnn3)
        cnn5_dispersions.append(data_cnn5)
        real_amplitude_dispersions.append(data_real_amplitude)
        cnn3_estimated_dispersions.append(data_cnn3_estimate)
        cnn5_estimated_dispersions.append(data_cnn5_estimate)

    of_color = '#9900ff'
    real_amplitude_color = "#FA3232"
    cnn3_color = "#B0B0B0"
    cnn5_color = "#1A1A1A"
    cnn3_estimated_color = 'darkorange'
    cnn5_estimated_color = 'deepskyblue'
    

    plt.plot(ocupacoes,of_dispersions, label="OF", marker='o', color = of_color)
    plt.plot(ocupacoes,cnn3_dispersions, label=r'CNN3 $\left(\hat{\tau} = \frac{A \tau}{\hat{A}_{CNN3}}\right)$', marker='*',linestyle='dashed', color = cnn3_color, zorder=5)
    plt.plot(ocupacoes,cnn5_dispersions, label=r'CNN5 $\left(\hat{\tau} = \frac{A \tau}{\hat{A}_{CNN5}}\right)$', marker='s', color = cnn5_color, linewidth=2)
    plt.plot(ocupacoes, real_amplitude_dispersions, label="Amplitude real", marker='o', color = real_amplitude_color)
    plt.plot(ocupacoes, cnn3_estimated_dispersions, label="CNN3", marker='s', markersize=6, color=cnn3_estimated_color, linewidth=2, zorder=4)
    plt.plot(ocupacoes, cnn5_estimated_dispersions, label="CNN5", marker='*', markersize=6, color=cnn5_estimated_color, linestyle='dashed', linewidth=1.5, zorder=5)
    plt.legend(loc='best')
    plt.xlabel('Ocupação')
    plt.ylabel('Dispersão fase estimada (ns)')
    plt.title(r'Dispersão $\times$ Ocupação')
    plt.tight_layout()
    plt.show()


PlotFaseDispersionOcupacao()