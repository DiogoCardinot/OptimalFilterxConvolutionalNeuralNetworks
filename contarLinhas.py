
import os
import numpy as np

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

N_JANELAMENTO=7

dataset_path = os.path.join(path,"ManipulacaoDados", "DadosConcatenados",f'janelamento_{N_JANELAMENTO}', "dataset_completo.npz")

#----------------------------- Data loading -------------------------------------------
data = np.load(dataset_path)

print(len(data['MatrizAmostras']))