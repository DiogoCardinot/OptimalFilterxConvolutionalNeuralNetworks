import numpy as np
import os

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7

base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")

def PlotFaseDispersionOcupacao():
    for ocupacao in ocupacoes:
        # PATHS
        of_path = os.path.join(dataset_path, "FiltroOtimo", "Dados", "OF_ErrorxRealPhase", f'janelamento_{n_janelamento}',f'errorxreal_{ocupacao}.npz')
        cnn3_path = os.path.join(dataset_path, "FiltroOtimo", "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_3', f'errorxreal_{ocupacao}.npz')
        cnn5_path = os.path.join(dataset_path, "FiltroOtimo", "Dados", "CNN_ErrorxRealPhase", f'janelamento_{n_janelamento}', f'CNN_5', f'errorxreal_{ocupacao}.npz')
        real_amplitude_path = os.path.join(dataset_path, "FiltroOtimo", "Dados", "RealAmplitude_ErrorxRealPhase", f'errorxreal_{ocupacao}.npz')
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
        
