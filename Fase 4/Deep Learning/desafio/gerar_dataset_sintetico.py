# -*- coding: utf-8 -*-
"""Gera o dataset do desafio NovaCommerce — 100% sintético, sem credenciais.

Produz a mesma estrutura que o desafio espera:

    desafio-novacommerce-v1/
    ├── reviews.csv      review_id, texto, label            (ok | atencao | remover)
    ├── produtos/        imagens 224x224 RGB em 10 pastas   (um departamento cada)
    ├── metadata.csv     produto_id, imagem, departamento
    └── README.txt

Por que sintético? O script original (preparation_script_kaggle.py) depende de
duas bases do Kaggle e de credencial pessoal. Esta versão roda em qualquer
máquina, é reprodutível por semente e não tem restrição de licença — mantendo
o problema didaticamente equivalente: texto ruidoso em português com três
classes desbalanceadas e imagens com padrões visuais aprendíveis.

Uso:
    python gerar_dataset_sintetico.py                 # escala de estudo (rápido)
    python gerar_dataset_sintetico.py --completo      # escala do enunciado
    python gerar_dataset_sintetico.py --reviews 20000 --imagens 5000
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SEED = 42

DEPARTAMENTOS = [
    # (nome, cor base RGB, forma característica)
    ("eletronicos", (38, 70, 160), "retangulo"),
    ("moda", (196, 62, 122), "triangulo"),
    ("casa", (176, 120, 50), "casa"),
    ("esporte", (34, 140, 90), "circulo"),
    ("beleza", (214, 96, 160), "gota"),
    ("livros", (120, 74, 44), "livro"),
    ("brinquedos", (232, 160, 40), "estrela"),
    ("automotivo", (70, 78, 92), "roda"),
    ("mercado", (200, 60, 48), "sacola"),
    ("pet", (96, 150, 190), "pata"),
]

# ---------------------------------------------------------------- reviews ---
PRODUTOS = ["fone bluetooth", "tênis de corrida", "cafeteira", "camiseta básica",
            "notebook", "panela de pressão", "ração premium", "livro de bolso",
            "pneu aro 15", "batom matte", "boneca articulada", "monitor 24”"]

OK_TEMPLATES = [
    "Chegou antes do prazo, {p} bem embalado. Recomendo.",
    "O {p} atendeu o que eu esperava, custo-benefício justo.",
    "Uso o {p} há duas semanas e até agora sem problema nenhum.",
    "Entrega rápida e o {p} veio certinho, igual ao anúncio.",
    "Produto bom, {p} de qualidade. Compraria de novo.",
    "Gostei bastante do {p}, principalmente do acabamento.",
    "Veio na data combinada. O {p} funciona bem para o preço.",
    "Segunda compra do mesmo vendedor, o {p} nunca decepcionou.",
    "Simples e eficiente, o {p} resolve o que promete.",
    "Nota 8 para o {p}: nada excepcional, mas cumpre o papel.",
]

ATENCAO_TEMPLATES = [
    "O {p} veio com um arranhão, mas o suporte respondeu rápido... acho.",
    "Não sei se recomendo o {p}, funcionou por uns dias e depois travou.",
    "Demorou 20 dias para chegar e o {p} veio sem a nota fiscal.",
    "O {p} é ok, mas o vendedor some quando você pede troca.",
    "Comprei o {p} achando que era original, tenho minhas dúvidas.",
    "Atenção: o {p} não é compatível com o que está descrito no anúncio.",
    "Preço subiu no dia seguinte, parece propaganda enganosa com o {p}.",
    "O {p} chegou trocado e ninguém responde o chamado até agora.",
    "Cuidado com a voltagem do {p}, quase queimei o meu.",
    "Foto do anúncio não corresponde ao {p} que veio na caixa.",
]

REMOVER_TEMPLATES = [
    "Vendedor picareta, esse {p} é uma porcaria, não caiam nessa!!!",
    "GANHE DINHEIRO EM CASA! acesse meu perfil, não perca tempo com {p}",
    "esse {p} é lixo e quem comprou é otário, fujam dessa loja",
    "COMPRE NO MEU SITE, MUITO MAIS BARATO QUE ESSE {p} AQUI, chama no zap",
    "review paga, ignorem, esse {p} é fraude pura, denuncio todo mundo",
    "!!!! PROMOÇÃO IMPERDÍVEL !!!! clique no link da bio, {p} pela metade",
    "o {p} é uma vergonha e o dono da loja é um golpista safado",
    "SIGA MEU PERFIL PARA CUPONS, esqueça esse {p} caro demais",
    "não comprem, loja de bandido, esse {p} nem chegou e me xingaram",
    "spam spam spam compre {p} no atacado direto comigo, sem taxa",
]

RUIDOS = ["", "", "", " ", "!!", "...", " kkk", " :)", " ???"]


def _ruidificar(texto: str, rnd: random.Random) -> str:
    t = texto + rnd.choice(RUIDOS)
    if rnd.random() < 0.12:
        t = t.upper()
    if rnd.random() < 0.10:
        t = t.replace("ã", "a").replace("ç", "c").replace("é", "e")
    if rnd.random() < 0.08:
        t = t.replace(" ", "  ", 1)
    return t.strip()


def gerar_reviews(n: int, destino: Path, rnd: random.Random) -> dict:
    # desbalanceamento proposital, como no enunciado (~15% inadequado)
    pesos = {"ok": 0.85, "atencao": 0.10, "remover": 0.05}
    bancos = {"ok": OK_TEMPLATES, "atencao": ATENCAO_TEMPLATES,
              "remover": REMOVER_TEMPLATES}
    labels = list(pesos)
    contagem = {k: 0 for k in labels}
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["review_id", "texto", "label"])
        for i in range(n):
            label = rnd.choices(labels, weights=[pesos[k] for k in labels])[0]
            texto = rnd.choice(bancos[label]).format(p=rnd.choice(PRODUTOS))
            w.writerow([f"rv_{i:06d}", _ruidificar(texto, rnd), label])
            contagem[label] += 1
    return contagem


# ---------------------------------------------------------------- imagens ---
def _desenhar_forma(d: ImageDraw.ImageDraw, forma: str, cx: int, cy: int,
                    r: int, cor: tuple, rnd: random.Random) -> None:
    if forma == "retangulo":
        d.rectangle([cx - r, cy - int(r * 0.65), cx + r, cy + int(r * 0.65)], fill=cor)
        d.rectangle([cx - int(r * 0.5), cy + int(r * 0.65), cx + int(r * 0.5),
                     cy + int(r * 0.85)], fill=cor)
    elif forma == "triangulo":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=cor)
    elif forma == "casa":
        d.polygon([(cx, cy - r), (cx - r, cy), (cx + r, cy)], fill=cor)
        d.rectangle([cx - int(r * 0.7), cy, cx + int(r * 0.7), cy + r], fill=cor)
    elif forma == "circulo":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cor)
        d.ellipse([cx - int(r * 0.4), cy - int(r * 0.4), cx + int(r * 0.4),
                   cy + int(r * 0.4)], fill=(250, 250, 250))
    elif forma == "gota":
        d.ellipse([cx - r, cy - int(r * 0.7), cx + r, cy + r], fill=cor)
        d.polygon([(cx, cy - r * 1.5), (cx - int(r * 0.5), cy), (cx + int(r * 0.5), cy)], fill=cor)
    elif forma == "livro":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=cor)
        d.line([(cx, cy - r), (cx, cy + r)], fill=(250, 250, 250), width=max(2, r // 12))
        for k in range(3):
            yy = cy - int(r * 0.4) + k * int(r * 0.4)
            d.line([(cx + int(r * 0.15), yy), (cx + int(r * 0.8), yy)],
                   fill=(250, 250, 250), width=max(1, r // 22))
    elif forma == "estrela":
        pts = []
        for k in range(10):
            ang = math.pi / 2 + k * math.pi / 5
            raio = r if k % 2 == 0 else r * 0.45
            pts.append((cx + raio * math.cos(ang), cy - raio * math.sin(ang)))
        d.polygon(pts, fill=cor)
    elif forma == "roda":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cor)
        d.ellipse([cx - int(r * 0.45), cy - int(r * 0.45), cx + int(r * 0.45),
                   cy + int(r * 0.45)], fill=(240, 240, 240))
        for k in range(6):
            ang = k * math.pi / 3
            d.line([(cx, cy), (cx + r * math.cos(ang), cy + r * math.sin(ang))],
                   fill=(240, 240, 240), width=max(2, r // 14))
    elif forma == "sacola":
        d.rectangle([cx - r, cy - int(r * 0.55), cx + r, cy + r], fill=cor)
        d.arc([cx - int(r * 0.55), cy - r, cx + int(r * 0.55), cy + int(r * 0.2)],
              200, 340, fill=cor, width=max(3, r // 10))
    elif forma == "pata":
        d.ellipse([cx - int(r * 0.62), cy - int(r * 0.2), cx + int(r * 0.62), cy + r], fill=cor)
        for dx in (-0.72, -0.24, 0.24, 0.72):
            d.ellipse([cx + int(r * dx) - int(r * 0.22), cy - int(r * 0.85),
                       cx + int(r * dx) + int(r * 0.22), cy - int(r * 0.32)], fill=cor)


def gerar_imagens(n: int, pasta: Path, meta_csv: Path, rnd: random.Random) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    por_dep = max(1, n // len(DEPARTAMENTOS))
    contagem = {}
    with open(meta_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["produto_id", "imagem", "departamento"])
        idx = 0
        for dep, cor_base, forma in DEPARTAMENTOS:
            dpasta = pasta / dep
            dpasta.mkdir(exist_ok=True)
            for k in range(por_dep):
                fundo = tuple(rnd.randint(228, 252) for _ in range(3))
                img = Image.new("RGB", (224, 224), fundo)
                d = ImageDraw.Draw(img)
                # variação de cor dentro do departamento (tom, não matiz)
                cor = tuple(max(0, min(255, c + rnd.randint(-28, 28))) for c in cor_base)
                cx = 112 + rnd.randint(-20, 20)
                cy = 112 + rnd.randint(-20, 20)
                r = rnd.randint(58, 82)
                _desenhar_forma(d, forma, cx, cy, r, cor, rnd)
                # ruído leve: manchas de fundo e desfoque ocasional
                for _ in range(rnd.randint(0, 5)):
                    x0, y0 = rnd.randint(0, 210), rnd.randint(0, 210)
                    s = rnd.randint(4, 14)
                    tom = tuple(max(0, min(255, c + rnd.randint(-18, 18))) for c in fundo)
                    d.ellipse([x0, y0, x0 + s, y0 + s], fill=tom)
                if rnd.random() < 0.25:
                    img = img.filter(ImageFilter.GaussianBlur(radius=rnd.uniform(0.3, 0.9)))
                nome = f"{dep}_{k:05d}.png"
                img.save(dpasta / nome)
                w.writerow([f"pd_{idx:06d}", f"{dep}/{nome}", dep])
                idx += 1
            contagem[dep] = por_dep
    return contagem


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", type=int, default=8000, help="nº de reviews (padrão 8000)")
    ap.add_argument("--imagens", type=int, default=3000, help="nº de imagens (padrão 3000)")
    ap.add_argument("--completo", action="store_true",
                    help="escala do enunciado: 40000 reviews e 15000 imagens")
    ap.add_argument("--saida", default="desafio-novacommerce-v1")
    ap.add_argument("--zip", action="store_true", help="compacta ao final")
    args = ap.parse_args()

    if args.completo:
        args.reviews, args.imagens = 40000, 15000

    rnd = random.Random(SEED)
    destino = Path(args.saida)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    print(f"Gerando {args.reviews} reviews...")
    cont_rv = gerar_reviews(args.reviews, destino / "reviews.csv", rnd)
    print("  distribuição:", cont_rv)

    print(f"Gerando {args.imagens} imagens 224x224 em {len(DEPARTAMENTOS)} departamentos...")
    cont_img = gerar_imagens(args.imagens, destino / "produtos", destino / "metadata.csv", rnd)
    print(f"  {sum(cont_img.values())} imagens ({list(cont_img.values())[0]} por departamento)")

    (destino / "README.txt").write_text(textwrap.dedent(f"""\
        NovaCommerce — dataset sintético do desafio
        ===========================================
        Gerado por gerar_dataset_sintetico.py (semente {SEED}).

        reviews.csv    {args.reviews} linhas: review_id, texto, label
                       labels: {cont_rv}
        produtos/      {sum(cont_img.values())} imagens 224x224 RGB em 10 departamentos
        metadata.csv   produto_id, imagem, departamento

        Observações para quem for usar:
        - Os dados são sintéticos e existem para o exercício de modelagem; não
          representam clientes, produtos ou opiniões reais.
        - As classes de review são desbalanceadas de propósito (~15% inadequado),
          para exigir tratamento explícito (class weights, focal loss ou reamostragem).
        - As imagens têm forma e paleta características por departamento, com
          ruído, deslocamento e desfoque — separáveis, mas não triviais.
        - Para a escala completa do enunciado: --completo
        """), encoding="utf-8")

    if args.zip:
        alvo = f"{args.saida}.zip"
        print(f"Compactando {alvo}...")
        with zipfile.ZipFile(alvo, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(destino.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(destino.parent))
    print("Pronto:", destino.resolve())


if __name__ == "__main__":
    main()
