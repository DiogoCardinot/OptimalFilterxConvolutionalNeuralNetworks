import subprocess
import sys
import os
import time

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

for i, script in enumerate(scripts):
    script_path = os.path.join(path, script)
    print(f"\n[{i+1}/{len(scripts)}] Iniciando {script}...")
    print(f"  Horário de início: {time.strftime('%d/%m/%Y %H:%M:%S')}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=path
    )

    elapsed = time.time() - start
    horas = int(elapsed // 3600)
    minutos = int((elapsed % 3600) // 60)
    segundos = int(elapsed % 60)

    if result.returncode == 0:
        print(f"  {script} concluído com sucesso!")
    else:
        print(f"  ERRO em {script} (código {result.returncode}) — continuando para o próximo...")

    print(f"  Tempo: {horas:02d}h {minutos:02d}m {segundos:02d}s")
    print(f"  Horário de fim: {time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("-" * 60)

print("\n=== TODOS OS SCRIPTS FINALIZADOS ===")
print(f"Horário final: {time.strftime('%d/%m/%Y %H:%M:%S')}")