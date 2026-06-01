import numpy as np
import os

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7

base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

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
        real_amplitude_path = os.path.join(dataset_path, "FiltroOtimo", f"FaseEstimada_RealAmplitude",  f'phase_real_amplitude_occupation_{ocupacao}.npz' )
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



