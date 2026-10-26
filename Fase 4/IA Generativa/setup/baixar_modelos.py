# -*- coding: utf-8 -*-
"""Pre-download de TODOS os modelos usados nos hands-on (aulas 01 a 06).

Rode com DIAS de antecedencia da gravacao — nunca dependa da rede no dia.
Download total: ~18 GB no cache do Hugging Face (%USERPROFILE%\\.cache\\huggingface).

Uso:
    python setup\\baixar_modelos.py            # baixa tudo
    python setup\\baixar_modelos.py --aula 02  # baixa so o necessario da aula 02
"""
import argparse
import time

from huggingface_hub import snapshot_download

# (modelo, tamanho aproximado, aulas em que e usado)
MODELOS = [
    ("pierreguillou/gpt2-small-portuguese",                        "~510 MB", ["01"]),
    ("neuralmind/bert-base-portuguese-cased",                      "~440 MB", ["01"]),
    ("Helsinki-NLP/opus-mt-tc-big-en-pt",                          "~950 MB", ["01", "05"]),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2","~470 MB", ["02"]),
    ("Qwen/Qwen2.5-1.5B-Instruct",                                 "~3.1 GB", ["02", "03", "04", "05", "06"]),
    ("Qwen/Qwen2.5-0.5B-Instruct",                                 "~1.0 GB", ["fallback (velocidade/VRAM)"]),
    ("Qwen/Qwen2.5-3B-Instruct",                                   "~6.2 GB", ["04 (plano B do agente)"]),
    ("sentence-transformers/clip-ViT-B-32",                        "~600 MB", ["05"]),
    ("sentence-transformers/clip-ViT-B-32-multilingual-v1",        "~450 MB", ["05"]),
    ("Salesforce/blip-image-captioning-base",                      "~990 MB", ["05"]),
]

# Arquivos de peso desnecessarios (ex.: versoes .h5/.msgpack duplicadas)
IGNORAR = ["*.h5", "*.msgpack", "*.tflite", "*.onnx", "*openvino*"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aula", help="baixar apenas os modelos de uma aula (ex.: 02)")
    args = parser.parse_args()

    fila = [
        (repo, tam, aulas)
        for repo, tam, aulas in MODELOS
        if not args.aula or any(args.aula in a for a in aulas)
    ]
    print(f"{len(fila)} modelo(s) na fila.\n")

    inicio = time.perf_counter()
    for i, (repo, tam, aulas) in enumerate(fila, 1):
        print(f"[{i}/{len(fila)}] {repo}  ({tam}; aulas {', '.join(aulas)})")
        snapshot_download(repo, ignore_patterns=IGNORAR)
        print("        ... ok\n")

    print("=" * 60)
    print(f"Concluido em {time.perf_counter() - inicio:.0f} s. Cache HF pronto para a gravacao.")


if __name__ == "__main__":
    main()
