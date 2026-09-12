# -*- coding: utf-8 -*-
"""Smoke test do ambiente — Redes Neurais e Deep Learning. Rode DEPOIS de instalar.ps1.

Esperado: todas as linhas com [OK]; em maquina com GPU NVIDIA,
"CUDA disponivel: True". Qualquer [FALHOU] precisa ser resolvido
antes de gravar (veja Troubleshooting no README.md).
"""
import importlib
import sys

print(f"Python: {sys.version.split()[0]} ({sys.executable})")
print("=" * 60)

PACOTES = [
    ("torch", "torch"),
                                ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("onnxruntime", "onnxruntime"),
]

falhas = []
for modulo, nome_pip in PACOTES:
    try:
        m = importlib.import_module(modulo)
        versao = getattr(m, "__version__", "?")
        print(f"[OK]     {nome_pip:22s} {versao}")
    except Exception as e:  # noqa: BLE001 - queremos reportar qualquer falha
        print(f"[FALHOU] {nome_pip:22s} {type(e).__name__}: {e}")
        falhas.append(nome_pip)

print("=" * 60)
try:
    import torch

    print(f"CUDA disponivel: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")
    else:
        print("(Sem GPU: os hands-on funcionam em CPU, apenas mais lentos.)")
except Exception as e:  # noqa: BLE001
    print(f"[FALHOU] torch/CUDA: {e}")
    falhas.append("torch")

print("=" * 60)
if falhas:
    print(f"AMBIENTE COM PROBLEMAS: {', '.join(falhas)}")
    sys.exit(1)
print("AMBIENTE OK - abra os notebooks com: jupyter lab")
