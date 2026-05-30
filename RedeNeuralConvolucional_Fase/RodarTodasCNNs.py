import subprocess
import sys
import os
import time
from datetime import datetime
import pickle

scripts = [
    "CNN1_KFold.py",
    "CNN2_KFold.py",
    "CNN3_KFold.py",
    "CNN4_KFold.py",
    "CNN5_KFold.py",
    "CNN6_KFold.py",
    "CNN7_KFold.py",
    "CNN8_KFold.py",
]

path = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("INICIANDO EXECUÇÃO DE TODOS OS CNNs")
print("=" * 60)


cnn_scripts        = []
cnn_tempo_inicio   = []
cnn_tempo_fim      = []
cnn_tempo_segundos = []
cnn_tempo_formatado= []
cnn_sucesso        = []
cnn_returncode     = []

tempo_inicial_geral = datetime.now()
for i, script in enumerate(scripts):
    script_path = os.path.join(path, script)
    print(f"\n[{i+1}/{len(scripts)}] Iniciando {script}...")
    tempo_inicio = datetime.now()
    print(f"  Horário de início: {tempo_inicio.strftime('%d/%m/%Y %H:%M:%S')}")
    
    start = time.time()
    result = subprocess.run([sys.executable, script_path], cwd=path)
    elapsed = time.time() - start
    tempo_fim = datetime.now()

    horas = int(elapsed // 3600)
    minutos = int((elapsed % 3600) // 60)
    segundos = int(elapsed % 60)
    sucesso = result.returncode == 0

    cnn_scripts.append(script)
    cnn_tempo_inicio.append(tempo_inicio.strftime('%d/%m/%Y %H:%M:%S'))
    cnn_tempo_fim.append(tempo_fim.strftime('%d/%m/%Y %H:%M:%S'))
    cnn_tempo_segundos.append(round(elapsed, 2))
    cnn_tempo_formatado.append(f'{horas:02d}:{minutos:02d}:{segundos:02d}')
    cnn_sucesso.append(sucesso)
    cnn_returncode.append(result.returncode)

    status = "concluído com sucesso!" if sucesso else f"ERRO (código {result.returncode})"
    print(f"  {script} {status}")
    print(f"  Tempo: {horas:02d}h {minutos:02d}m {segundos:02d}s")
    print(f"  Horário de fim: {tempo_fim.strftime('%d/%m/%Y %H:%M:%S')}")
    print("-" * 60)



print("\n=== TODOS OS SCRIPTS FINALIZADOS ===")
tempo_final_geral = datetime.now()
total_elapsed = (tempo_final_geral - tempo_inicial_geral).total_seconds()                                                
tempo_medio   = total_elapsed / len(scripts)
scripts_com_erro = [s for s, ok in zip(cnn_scripts, cnn_sucesso) if not ok]

def formatar_tempo(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def montar_logs(cnn_scripts, cnn_tempo_inicio, cnn_tempo_fim,
                cnn_tempo_segundos, cnn_tempo_formatado,
                cnn_sucesso, cnn_returncode,
                tempo_inicial_geral, tempo_final_geral,
                total_elapsed, tempo_medio, scripts_com_erro):
    logs = {}

    for i, script in enumerate(cnn_scripts):
        logs[f'CNN{i+1}'] = {
            'script':          script,
            'tempo_inicio':    cnn_tempo_inicio[i],
            'tempo_fim':       cnn_tempo_fim[i],
            'tempo_segundos':  cnn_tempo_segundos[i],
            'tempo_formatado': cnn_tempo_formatado[i],
            'sucesso':         cnn_sucesso[i],
            'returncode':      cnn_returncode[i],
        }

    logs['geral'] = {
        'hora_inicial':          tempo_inicial_geral.strftime('%d/%m/%Y %H:%M:%S'),
        'hora_final':            tempo_final_geral.strftime('%d/%m/%Y %H:%M:%S'),
        'tempo_total_segundos':  round(total_elapsed, 2),
        'tempo_total_formatado': formatar_tempo(total_elapsed),
        'tempo_medio_segundos':  round(tempo_medio, 2),
        'tempo_medio_formatado': formatar_tempo(tempo_medio),
        'scripts_com_erro':      scripts_com_erro if scripts_com_erro else ['nenhum'],
        'todos_ok':              len(scripts_com_erro) == 0,
    }

    return logs


logs = montar_logs(
    cnn_scripts, cnn_tempo_inicio, cnn_tempo_fim,
    cnn_tempo_segundos, cnn_tempo_formatado,
    cnn_sucesso, cnn_returncode,
    tempo_inicial_geral, tempo_final_geral,
    total_elapsed, tempo_medio, scripts_com_erro
)

output_path=os.path.join(path, "TimeLogs")
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, 'time_logs.pkl')

with open(output_file, 'wb') as f:
    pickle.dump(logs, f)


# ----------------------------- Resumo final -------------------------------------------
print("\n=== TODOS OS SCRIPTS FINALIZADOS ===")
print(f"Horário final:  {tempo_final_geral.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Tempo total:    {formatar_tempo(total_elapsed)}")
print(f"Tempo médio:    {formatar_tempo(tempo_medio)}")
if scripts_com_erro:
    print(f"Scripts com erro: {scripts_com_erro}")
else:
    print("Todos os scripts concluídos com sucesso!")
print(f"Log salvo em: {output_file}")