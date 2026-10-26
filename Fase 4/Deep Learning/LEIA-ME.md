# Redes Neurais e Deep Learning — FIAP Pós-Tech

Prof. Flavio Padilha · AI Engineering · 2026

## O que tem aqui

```
aula-01-fundamentos/
  apresentacao/aula-01.pptx    slides da aula
  apostila/aula-01.docx        material teórico completo
  handson/hands-on.ipynb       notebook do experimento (já executado)
  handson/README.md            como rodar e o que esperar
... (idem para as 8 aulas)

desafio/                       o desafio integrador NovaCommerce
setup/                         instalação do ambiente
requirements.txt
```

## Como começar

1. **Instale o ambiente** (uma vez só):

   ```powershell
   powershell -ExecutionPolicy Bypass -File setup\instalar.ps1
   ```

   Sem GPU também funciona — o notebook mais pesado roda em cerca de um
   minuto em CPU.

2. **Abra o notebook da aula** com o kernel do venv:

   ```
   jupyter lab
   ```

   Os notebooks vêm com as saídas da execução de referência. Rode
   "Restart & Run All" para reproduzir na sua máquina e comparar.

3. **Leia antes de rodar.** Cada notebook é um estudo experimental: a
   pergunta e a hipótese vêm ANTES do código. Leia o enunciado, escreva o
   que você espera encontrar e só então execute. Depois mude uma variável
   por vez — assim as suas conclusões se sustentam.

## O desafio

O desafio NovaCommerce vale 60% da nota. O dataset é gerado localmente:

```bash
python desafio/gerar_dataset_sintetico.py --zip
```

Leia `desafio/desafio.docx` para o enunciado completo.

## Ordem sugerida

As aulas se apoiam umas nas outras: fundamentos (1–2) → visão (3–4) →
sequências (5–6) → treino e produção (7–8). O desafio integra as quatro
frentes, então vale começar a pensar nele a partir da aula 4.
