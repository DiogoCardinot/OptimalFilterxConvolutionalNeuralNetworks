import os
import numpy as np


root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)
base_path = os.path.dirname(os.path.dirname(path))

ocupacoes = [10,50,80,100]
n_janelamento = 7

def DefinePath(ocupacao):
    if ocupacao == 10 or ocupacao == 50:
        CNN = 5
        
    elif ocupacao==80 or ocupacao == 100:
        CNN=3
    
    CNN_data_path = os.path.join(base_path, f'RedeNeuralConvolucional',f'CNN_{CNN}', f"results_ocupacao_{ocupacao}.npz")
    cnn_type = f'CNN-{CNN}'
    CNN_data = np.load(CNN_data_path)

    return CNN_data_path, CNN_data, cnn_type

def ImprimeMetricas():
    of_data_parcial = os.path.join(base_path, "FiltroOtimo", "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
    for ocupacao in ocupacoes:
        OF_data_path = os.path.join(of_data_parcial, f"results_occupation_{ocupacao}.npz")
        OF_data = np.load(OF_data_path)
        _, CNN_data, cnn_type = DefinePath(ocupacao)
        print(f"\nComparacao das Arquiteturas de CNN - Ocupacao {ocupacao}\n")
        print("| Metrica       | RMS      | R^2       | MAE      | MedAE    |")
        print("|" + "-"*15 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
        print(f"| OF            | {OF_data['rms']:.6f} | {OF_data['r2']:.6f} | {OF_data['mae']:.6f} | {OF_data['medae']:.6f} |")
        print(f"| {cnn_type}           | {CNN_data['rms']:.6f} | {CNN_data['r2']:.6f} | {CNN_data['mae']:.6f} | {CNN_data['medae']:.6f} |")
        print(100*"=")

ImprimeMetricas()