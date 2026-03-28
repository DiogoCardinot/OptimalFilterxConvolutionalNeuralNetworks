import numpy as np
import os

n_janelamento = 9
pedestal = 30
ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]

caminho_arquivo = os.path.abspath(__file__)
diretorio_pai = os.path.dirname(caminho_arquivo)
pasta_dados = os.path.join(diretorio_pai,"../", "Dados_Ocupacoes")
pasta_saida = os.path.join(diretorio_pai, "DadosPorOcupacao",f"janelamento_{n_janelamento}")

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

def montarMatrizSinaisEAmplitude(nome_arquivo_amostras, n_janelamento):
    dados_amostras = np.genfromtxt(nome_arquivo_amostras, delimiter=",", skip_header=1)

    num_linhas = len(dados_amostras) - (n_janelamento - 1)
    matriz_amostras = np.zeros((num_linhas, n_janelamento))
    for i in range(num_linhas):
        inicio = i
        fim = i + n_janelamento
        matriz_amostras[i] = dados_amostras[inicio:fim, 1] - pedestal

    indice_central = n_janelamento // 2
    amplitude_real = np.zeros(num_linhas)
    phase_real = np.zeros(num_linhas)
    for i in range(num_linhas):
        amplitude_real[i] = dados_amostras[i + indice_central, 2]
        phase_real[i] = dados_amostras[i+indice_central, 3]

    return matriz_amostras, amplitude_real, phase_real

for ocupacao in ocupacoes:
    nome_arquivo_amostras_ocupacao = os.path.join(pasta_dados, f"OC_{ocupacao}.txt")

    if not os.path.exists(nome_arquivo_amostras_ocupacao):
        print(f" Arquivo nao encontrado para ocupacao {ocupacao}, pulando...")
        continue

    matriz_amostras, amplitude_real, phase_real = montarMatrizSinaisEAmplitude(nome_arquivo_amostras_ocupacao, n_janelamento)
    nome_arquivo_saida = os.path.join(pasta_saida, f"dataset_ocup_{ocupacao}.npz")
    np.savez_compressed(nome_arquivo_saida,
                        matriz_amostras=matriz_amostras,
                        amplitude_real=amplitude_real, phase_real = phase_real)

    print(f"Ocupacao {ocupacao}: {matriz_amostras.shape[0]} amostras salvas em {nome_arquivo_saida}\n")
