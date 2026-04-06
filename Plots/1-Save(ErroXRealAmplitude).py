import numpy as np
import os
import matplotlib.pyplot as plt

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]
n_janelamento = 7

CNN = 8

def ErrorOF():
    print("OF")
    output = os.path.join(path, "Dados", "OF_ErrorxRealAmplitude", f'janelamento_{n_janelamento}')
    os.makedirs(output, exist_ok=True)
    
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")
    for ocupacao in ocupacoes:
        output_file = os.path.join(output, f'errorxreal_{ocupacao}.npz')
        
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Arquivo antigo removido: {output_file}")
            except PermissionError:
                print(f"Não foi possível remover {output_file}. Verifique se o arquivo não está aberto em outro programa.")
                continue
        
        error_of_path = os.path.join(dataset_path,"FiltroOtimo", f'AmplitudeEstimada_OF',f'janelamento_{n_janelamento}', f'results_occupation_{ocupacao}.npz')
        error_of_data = np.load(error_of_path)
        error_of = error_of_data['error']
        real_amplitude = error_of_data['real_amplitude']

        min_amplitude = min(real_amplitude)
        max_amplitude = max(real_amplitude)
        
        n_bins = 20
        bins = np.linspace(min_amplitude, max_amplitude, n_bins + 1)
        
        erros_por_intervalo = {}
        
        for i in range(len(bins) - 1):
            chave = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
            erros_por_intervalo[chave] = []
        
        for amp, erro in zip(real_amplitude, error_of):
            for i in range(len(bins) - 1):
                if bins[i] <= amp < bins[i+1]:
                    chave = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
                    erros_por_intervalo[chave].append(erro)
                    break
        
        stats_por_intervalo = {}
        for chave, erros in erros_por_intervalo.items():
            if erros:
                stats_por_intervalo[chave] = {
                    'media': np.mean(erros),
                    'mediana': np.median(erros),
                    'std': np.std(erros),
                    'min': np.min(erros),
                    'max': np.max(erros),
                    'n_amostras': len(erros)
                }
        
        try:
            np.savez_compressed(output_file, stats_por_intervalo=stats_por_intervalo, erros_por_intervalo= erros_por_intervalo)
            print(f"Arquivo salvo com sucesso: {output_file}")
        except PermissionError as e:
            print(f"Erro ao salvar {output_file}: {e}")
            print("Verifique se o arquivo não está aberto em outro programa (Excel, VS Code, etc.)")
            # Tentar salvar com nome alternativo
            alt_file = os.path.join(output, f'errorxreal_{ocupacao}_temp.npz')
            np.savez_compressed(alt_file, stats_por_intervalo=stats_por_intervalo, erros_por_intervalo= erros_por_intervalo)
            print(f"Arquivo salvo como alternativa: {alt_file}")
    print(50*"=")

def ErrorCNN():
    print("CNN")
    output = os.path.join(path, "Dados", "CNN_ErrorxRealAmplitude", f'janelamento_{n_janelamento}', f'CNN_{CNN}')
    os.makedirs(output, exist_ok=True)
    
    base_path = os.path.dirname(os.path.dirname(path))
    dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks")
    for ocupacao in ocupacoes:
        output_file = os.path.join(output, f'errorxreal_{ocupacao}.npz')
        
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Arquivo antigo removido: {output_file}")
            except PermissionError:
                print(f"Não foi possível remover {output_file}. Verifique se o arquivo não está aberto em outro programa.")
                continue
        
        error_cnn_path = os.path.join(dataset_path, "RedeNeuralConvolucional",f'CNN_{CNN}', f'results_ocupacao_{ocupacao}.npz')
        error_cnn_data = np.load(error_cnn_path)
        error_cnn = error_cnn_data['error']
        real_amplitude = error_cnn_data['real_amplitude']
    
        min_amplitude = min(real_amplitude)
        max_amplitude = max(real_amplitude)
        
        n_bins = 20
        bins = np.linspace(min_amplitude, max_amplitude, n_bins + 1)
        
        erros_por_intervalo = {}
        
        for i in range(len(bins) - 1):
            chave = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
            erros_por_intervalo[chave] = []
        
        for amp, erro in zip(real_amplitude, error_cnn):
            for i in range(len(bins) - 1):
                if bins[i] <= amp < bins[i+1]:
                    chave = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
                    erros_por_intervalo[chave].append(erro)
                    break
        
        stats_por_intervalo = {}
        for chave, erros in erros_por_intervalo.items():
            if erros:
                stats_por_intervalo[chave] = {
                    'media': np.mean(erros),
                    'mediana': np.median(erros),
                    'std': np.std(erros),
                    'min': np.min(erros),
                    'max': np.max(erros),
                    'n_amostras': len(erros)
                }
        
        try:
            np.savez_compressed(output_file, stats_por_intervalo=stats_por_intervalo, erros_por_intervalo= erros_por_intervalo)
            print(f"Arquivo salvo com sucesso: {output_file}")
        except PermissionError as e:
            print(f"Erro ao salvar {output_file}: {e}")
            print("Verifique se o arquivo não está aberto em outro programa (Excel, VS Code, etc.)")
            alt_file = os.path.join(output, f'errorxreal_{ocupacao}_temp.npz')
            np.savez_compressed(alt_file, stats_por_intervalo=stats_por_intervalo, erros_por_intervalo= erros_por_intervalo)
            print(f"Arquivo salvo como alternativa: {alt_file}")
    print(50*"=")

# ErrorOF()
ErrorCNN()