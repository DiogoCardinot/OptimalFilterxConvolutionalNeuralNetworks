import numpy as np
import glob
import os

n_janelamento = 9

caminho_arquivo = os.path.abspath(__file__)
diretorio_pai = os.path.dirname(caminho_arquivo)

# Pasta onde estão os .npz individuais
pasta_saida = os.path.join(diretorio_pai, "DadosPorOcupacao", f'janelamento_{n_janelamento}')
padrao_busca = os.path.join(pasta_saida, "dataset_ocup_*.npz")

MatrizAmostras, AmplitudeReal, FaseReal, Ocupacao = [], [], [], []

# Loop pelos arquivos encontrados
for file in glob.glob(padrao_busca):
    ocupacao = int(file.split('_')[-1].split('.')[0])
    
    data = np.load(file)
    MatrizAmostras.append(data['matriz_amostras'])
    AmplitudeReal.append(data['amplitude_real'])
    FaseReal.append(data['phase_real'])
    Ocupacao.append(np.full(data['amplitude_real'].shape, ocupacao))

# Concatena tudo
MatrizAmostras = np.vstack(MatrizAmostras)
AmplitudeReal = np.concatenate(AmplitudeReal)
FaseReal= np.concatenate(FaseReal)
Ocupacao = np.concatenate(Ocupacao)

print(f"Tamanho total:\nMatriz Amostras: {MatrizAmostras.shape}\nAmplitude Real: {AmplitudeReal.shape}\nFase Real: {FaseReal.shape}\nOcupacoes: {Ocupacao.shape}")

unique, counts = np.unique(Ocupacao, return_counts=True)
for o, c in zip(unique, counts):
    print(f"Ocupacao {o}: {c} amostras")

# Salvar dataset completo
caminho_dataset_completo = os.path.join(diretorio_pai, "DadosConcatenados", f"janelamento_{n_janelamento}")
if not os.path.exists(caminho_dataset_completo):
    os.makedirs(caminho_dataset_completo)
    
caminho_dataset_completo_final = os.path.join(caminho_dataset_completo, "dataset_completo.npz")
if os.path.exists(caminho_dataset_completo_final):
    os.remove(caminho_dataset_completo_final)

np.savez_compressed(caminho_dataset_completo_final, MatrizAmostras=MatrizAmostras, AmplitudeReal=AmplitudeReal, FaseReal= FaseReal,  Ocupacao=Ocupacao)
print(f"Dataset combinado salvo como: {caminho_dataset_completo_final}")
