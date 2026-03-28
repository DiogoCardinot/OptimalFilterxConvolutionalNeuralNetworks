import numpy as np
import os

n_janelamento = 7
EPS = 1e-8

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

of_data = os.path.join(path, "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
a_tau_data = os.path.join(path,"A_tau_OF", f'janelamento_{n_janelamento}')

base_path = os.path.dirname(os.path.dirname(path))
cnn_data = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks","RedeNeuralConvolucional", "CNN_8")


output_folder = os.path.join(path, "FaseEstimada", f'janelamento_{n_janelamento}')
os.makedirs(output_folder, exist_ok=True)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]

results_summary = {
    'of': {},
    'cnn': {}
}

for ocupacao in ocupacoes:
    print(f"Ocupacao - {ocupacao}")
    of_amplitude_filepath = os.path.join(of_data, f'results_occupation_{ocupacao}.npz')
    cnn_amplitude_filepath = os.path.join(cnn_data, f'results_occupation_{ocupacao}.npz')
    a_tau_filepath = os.path.join(a_tau_data, f'results_occupation_{ocupacao}.npz')

    if not os.path.exists(a_tau_filepath):
        print(f"  ERRO: Arquivo A_tau não encontrado: {a_tau_filepath}")
        continue
    
    if not os.path.exists(of_amplitude_filepath):
        print(f"  ERRO: Arquivo OF amplitude não encontrado: {of_amplitude_filepath}")
        continue
    
    if not os.path.exists(cnn_amplitude_filepath):
        print(f"  AVISO: Arquivo CNN amplitude não encontrado: {cnn_amplitude_filepath}")
        print(f"  Continuando apenas com OF...")


    a_tau_data_file = np.load(a_tau_filepath)
    of_amp_data_file = np.load(of_amplitude_filepath)
    cnn_amp_data_file = np.load(cnn_amplitude_filepath)
    

    a_tau_estimated = a_tau_data_file['estimated_A_tau']
    a_tau_indices = a_tau_data_file['indices']
    real_phase = a_tau_data_file['real_phase']
    
    of_estimated_amplitude = of_amp_data_file['estimated_amplitude']
    of_indices = of_amp_data_file['indices']

    cnn_estimated_amplitude = cnn_amp_data_file['estimated_amplitude']
    

    # estimar a amplitude alinhando os indices 
