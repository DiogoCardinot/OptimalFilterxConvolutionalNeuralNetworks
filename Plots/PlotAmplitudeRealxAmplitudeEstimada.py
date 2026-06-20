import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

plt.rcParams['savefig.directory'] = os.path.dirname(path)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['agg.path.chunksize'] = 10000

def PlotAmplitudeRealAmplitudeEstimada():
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

    ocupacoes = [100]
    n_janelamento = 7

    data = os.path.join(dataset_path, f'FiltroOtimo', f'AmplitudeEstimada_OF', f'janelamento_{n_janelamento}')

    for ocupacao in ocupacoes:
        data_file = os.path.join(data, f'results_occupation_{ocupacao}.npz')
        data_load = np.load(data_file)

        real_amplitude = data_load['real_amplitude']
        estimated_amplitude = data_load['estimated_amplitude']

        plt.scatter(real_amplitude, estimated_amplitude, s=1, alpha=0.3)
        plt.ylabel(r'Estimated Amplitude OF', fontsize = 12)
        plt.xlabel(r'Real Amplitude', fontsize = 12)
        plt.title(rf'Estimated Amplitude OF $\times$ Real Amplitude - Occupancy {ocupacao}%', fontsize = 14)
        plt.tight_layout()
        plt.show()
    

PlotAmplitudeRealAmplitudeEstimada()