"""
Dataset Preparation Script — desafio-novacommerce-v1.zip
=========================================================
Consolidates two Kaggle sources into the synthetic NovaCommerce dataset
required by the FIAP "Redes Neurais e Deep Learning" challenge.

Sources
-------
1. olistbr/brazilian-ecommerce   (CC BY-NC-SA 4.0) → reviews.csv
2. paramaggarwal/fashion-product-images-small (MIT)  → produtos/ + metadata.csv

Usage
-----
    # 1. Install dependencies
    pip install kaggle pandas pillow tqdm

    # 2. Ensure ~/.kaggle/kaggle.json has valid credentials
    #    (https://www.kaggle.com/docs/api#authentication)

    # 3. Run
    python preparation_script.py [--output-dir ./desafio-novacommerce-v1]

Outputs
-------
    desafio-novacommerce-v1/
    ├── reviews.csv       (40 000 rows: review_id, texto, label)
    ├── produtos/          (15 000 images 224×224 in 10 department folders)
    ├── metadata.csv       (produto_id, imagem, departamento)
    └── README.txt

    desafio-novacommerce-v1.zip   (final distributable archive)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
REVIEWS_TARGET = 40_000
IMAGES_PER_DEPT = 1_500
NUM_DEPARTMENTS = 10
IMAGES_TARGET = IMAGES_PER_DEPT * NUM_DEPARTMENTS  # 15 000
IMAGE_SIZE = (224, 224)

REVIEW_LABEL_DIST = {
    "ok": 0.70,       # 28 000
    "atencao": 0.15,   # 6 000
    "remover": 0.15,   # 6 000
}

# Mapping: review_score → moderation label
SCORE_TO_LABEL = {
    5: "ok",
    4: "ok",
    3: "atencao",
    2: "remover",
    1: "remover",
}

# 10 NovaCommerce departments mapped from fashion-product-images categories
DEPARTMENTS: list[str] = [
    "roupas_femininas",
    "roupas_masculinas",
    "calcados",
    "acessorios",
    "bolsas_e_malas",
    "esportes",
    "beleza",
    "joias_e_relogios",
    "casa",
    "infantil",
]

# ---------------------------------------------------------------------------
# Category → department mapping
# Uses (masterCategory, subCategory, gender) from styles.csv
# ---------------------------------------------------------------------------


def _map_department(row: pd.Series) -> str | None:
    """Return a NovaCommerce department name or None if unmappable."""
    master = str(row.get("masterCategory", "")).strip().lower()
    sub = str(row.get("subCategory", "")).strip().lower()
    article = str(row.get("articleType", "")).strip().lower()
    gender = str(row.get("gender", "")).strip().lower()

    # Kids first (any category)
    if gender in ("boys", "girls", "unisex kids"):
        return "infantil"

    # Footwear
    if master == "footwear" or sub in ("shoes", "flip flops", "sandal"):
        return "calcados"

    # Sporting Goods
    if master == "sporting goods" or sub == "sporting goods":
        return "esportes"

    # Personal Care → beauty
    if master == "personal care" or sub in (
        "fragrance", "lips", "eyes", "nails", "skin care", "hair",
        "skin", "makeup", "bath and body",
    ):
        return "beleza"

    # Home
    if master == "home" or sub in ("home furnishing", "home decor"):
        return "casa"

    # Jewellery & Watches
    if sub in ("watches", "jewellery", "precious jewellery"):
        return "joias_e_relogios"

    # Bags & Luggage
    if sub in ("bags", "wallets", "clutches", "travel accessories"):
        return "bolsas_e_malas"

    # Accessories (belts, ties, eyewear, headwear, etc.)
    if master == "accessories" and sub not in (
        "watches", "jewellery", "precious jewellery", "bags", "wallets",
        "clutches", "travel accessories",
    ):
        return "acessorios"

    # Apparel by gender
    if master == "apparel":
        if gender == "women":
            return "roupas_femininas"
        if gender == "men":
            return "roupas_masculinas"
        # Fall back to feminine for unisex apparel
        return "roupas_femininas"

    return None


# ---------------------------------------------------------------------------
# Synthetic review augmentation (Portuguese templates)
# ---------------------------------------------------------------------------

_SPAM_TEMPLATES = [
    "Compre agora no site {site}, preços imbatíveis! Link: {link}",
    "PROMOÇÃO IMPERDÍVEL! Acesse {link} e ganhe desconto!",
    "Trabalhe de casa e ganhe R${val}/dia. Saiba mais em {link}",
    "Clique aqui para ganhar um iPhone grátis: {link}",
    "Ganhe dinheiro fácil! Acesse {link} agora mesmo!",
    "Não perca essa oportunidade única! {link}",
    "★★★ OFERTA EXCLUSIVA ★★★ Entre em {link}",
    "Vendo seguidores Instagram barato. Contato: {link}",
]

_OFFENSIVE_TEMPLATES = [
    "Produto horrível, empresa lixo. Não comprem!",
    "Pior compra da minha vida. Atendimento péssimo e desrespeitoso.",
    "Entrega atrasada, produto quebrado. Uma vergonha total.",
    "Vendedor mentiroso, produto falso. Fraude total!",
    "Nunca mais compro aqui. Produto não corresponde ao anúncio.",
    "Enganação pura. Produto veio completamente diferente.",
    "Atendimento ridículo, ninguém responde. Empresa de fundo de quintal.",
    "Produto já veio com defeito e se recusam a trocar. Absurdo!",
]

_OK_TEMPLATES = [
    "Produto chegou certinho, conforme o anunciado. Recomendo!",
    "Boa qualidade pelo preço. Entrega rápida.",
    "Gostei muito, superou minhas expectativas.",
    "Excelente custo-benefício. Vou comprar novamente.",
    "Produto muito bom, embalagem segura.",
    "Atendeu perfeitamente minha necessidade. Nota 10!",
    "Material de qualidade, bem acabado. Satisfeito.",
    "Entrega no prazo e produto conforme descrito.",
]

_ATENCAO_TEMPLATES = [
    "Produto ok, mas a embalagem poderia ser melhor.",
    "Demorou mais do que o esperado para chegar.",
    "Cor ligeiramente diferente da foto, mas funciona bem.",
    "Regular, esperava um pouco mais pela faixa de preço.",
    "Produto razoável. Nada excepcional, nada terrível.",
    "Achei o material meio frágil, mas serve para o propósito.",
    "Tamanho um pouco diferente do indicado na tabela.",
    "Funciona, mas o acabamento deixa a desejar em alguns detalhes.",
]


def _random_url() -> str:
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
    domains = ["bit.ly", "encurta.cc", "goo.gl", "tinyurl.com"]
    return f"https://{random.choice(domains)}/{h}"


def _generate_synthetic_review(label: str) -> str:
    if label == "remover":
        if random.random() < 0.4:
            tmpl = random.choice(_SPAM_TEMPLATES)
            return tmpl.format(
                site=f"loja{random.randint(1,999)}.com",
                link=_random_url(),
                val=random.choice([50, 100, 200, 500]),
            )
        return random.choice(_OFFENSIVE_TEMPLATES)
    if label == "atencao":
        return random.choice(_ATENCAO_TEMPLATES)
    return random.choice(_OK_TEMPLATES)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_kaggle_dataset(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset using the kaggle CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"⬇  Downloading {slug} → {dest}")
    ret = os.system(
        f'kaggle datasets download -d "{slug}" -p "{dest}" --unzip --quiet'
    )
    if ret != 0:
        sys.exit(f"ERROR: kaggle download failed for {slug} (exit {ret})")


# ---------------------------------------------------------------------------
# Reviews pipeline
# ---------------------------------------------------------------------------


def build_reviews(olist_dir: Path, output_dir: Path) -> None:
    """Build reviews.csv from Olist order reviews."""
    print("\n📝 Building reviews.csv …")

    reviews_path = olist_dir / "olist_order_reviews_dataset.csv"
    if not reviews_path.exists():
        # Try nested directory
        for p in olist_dir.rglob("olist_order_reviews_dataset.csv"):
            reviews_path = p
            break

    if not reviews_path.exists():
        sys.exit(f"ERROR: Cannot find olist_order_reviews_dataset.csv in {olist_dir}")

    df = pd.read_csv(reviews_path)
    print(f"   Loaded {len(df)} raw review rows")

    # Keep only rows with non-empty review text
    df = df.dropna(subset=["review_comment_message"])
    df = df[df["review_comment_message"].str.strip().astype(bool)]
    print(f"   {len(df)} rows with non-empty review text")

    # Assign label from score
    df["label"] = df["review_score"].map(SCORE_TO_LABEL)
    df = df.dropna(subset=["label"])

    # Deduplicate by text content
    df = df.drop_duplicates(subset=["review_comment_message"])
    print(f"   {len(df)} unique reviews after dedup")

    # Target counts per label
    target = {
        lbl: int(REVIEWS_TARGET * frac)
        for lbl, frac in REVIEW_LABEL_DIST.items()
    }

    rows: list[dict] = []
    rng = random.Random(SEED)

    for label, count in target.items():
        pool = df[df["label"] == label]["review_comment_message"].tolist()
        rng.shuffle(pool)

        if len(pool) >= count:
            selected = pool[:count]
        else:
            # Use all real reviews, fill remainder with synthetic
            selected = list(pool)
            remaining = count - len(selected)
            print(f"   ⚠ Label '{label}': {len(pool)} real reviews, generating {remaining} synthetic")
            for _ in range(remaining):
                selected.append(_generate_synthetic_review(label))

        for text in selected:
            rows.append({"texto": text, "label": label})

    rng.shuffle(rows)

    # Assign sequential review IDs
    out_rows = []
    for i, r in enumerate(rows, start=1):
        out_rows.append({
            "review_id": f"R{i:06d}",
            "texto": r["texto"],
            "label": r["label"],
        })

    out_path = output_dir / "reviews.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"   ✅ Wrote {len(out_df)} reviews → {out_path}")

    # Distribution summary
    dist = out_df["label"].value_counts()
    for lbl in ["ok", "atencao", "remover"]:
        print(f"      {lbl}: {dist.get(lbl, 0):,}")


# ---------------------------------------------------------------------------
# Images pipeline
# ---------------------------------------------------------------------------


def build_images_and_metadata(
    fashion_dir: Path, output_dir: Path
) -> None:
    """Build produtos/ images and metadata.csv from fashion product images."""
    print("\n🖼️  Building produtos/ and metadata.csv …")

    styles_path = fashion_dir / "styles.csv"
    if not styles_path.exists():
        for p in fashion_dir.rglob("styles.csv"):
            styles_path = p
            break

    if not styles_path.exists():
        sys.exit(f"ERROR: Cannot find styles.csv in {fashion_dir}")

    # styles.csv sometimes has bad lines; use on_bad_lines='skip'
    df = pd.read_csv(styles_path, on_bad_lines="skip")
    print(f"   Loaded {len(df)} product rows from styles.csv")

    # Find the images directory
    images_dir = styles_path.parent / "images"
    if not images_dir.exists():
        # Try parent
        for candidate in fashion_dir.rglob("images"):
            if candidate.is_dir():
                images_dir = candidate
                break

    if not images_dir.exists():
        sys.exit(f"ERROR: Cannot find images/ folder in {fashion_dir}")

    # Map each product to a department
    df["departamento"] = df.apply(_map_department, axis=1)
    df = df.dropna(subset=["departamento"])
    print(f"   {len(df)} products mapped to departments")

    # Show distribution
    dept_counts = df["departamento"].value_counts()
    for d in DEPARTMENTS:
        print(f"      {d}: {dept_counts.get(d, 0):,} available")

    # Check which images actually exist on disk
    df["image_file"] = df["id"].astype(str) + ".jpg"
    df["image_path"] = df["image_file"].apply(lambda f: images_dir / f)
    df["image_exists"] = df["image_path"].apply(lambda p: p.exists())
    df = df[df["image_exists"]]
    print(f"   {len(df)} products with existing image files")

    # Sample per department
    rng = random.Random(SEED)
    produtos_dir = output_dir / "produtos"
    metadata_rows: list[dict] = []
    produto_counter = 0

    for dept in DEPARTMENTS:
        dept_dir = produtos_dir / dept
        dept_dir.mkdir(parents=True, exist_ok=True)

        pool = df[df["departamento"] == dept]

        if len(pool) == 0:
            print(f"   ⚠ No images for department '{dept}' — skipping")
            continue

        # Sample with replacement if pool is too small
        if len(pool) >= IMAGES_PER_DEPT:
            sampled = pool.sample(n=IMAGES_PER_DEPT, random_state=SEED)
        else:
            print(
                f"   ⚠ Department '{dept}': only {len(pool)} images,"
                f" sampling with replacement to reach {IMAGES_PER_DEPT}"
            )
            sampled = pool.sample(
                n=IMAGES_PER_DEPT, replace=True, random_state=SEED
            )

        for _, row in tqdm(
            sampled.iterrows(),
            total=IMAGES_PER_DEPT,
            desc=f"   {dept}",
            leave=False,
        ):
            produto_counter += 1
            produto_id = f"P{produto_counter:06d}"
            out_filename = f"{produto_id}.jpg"
            out_path = dept_dir / out_filename

            try:
                img = Image.open(row["image_path"]).convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.LANCZOS)
                img.save(out_path, "JPEG", quality=90)
            except Exception as exc:
                print(f"   ⚠ Skipped {row['image_file']}: {exc}")
                continue

            metadata_rows.append({
                "produto_id": produto_id,
                "imagem": f"produtos/{dept}/{out_filename}",
                "departamento": dept,
            })

    # Write metadata.csv
    meta_path = output_dir / "metadata.csv"
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(meta_path, index=False)
    print(f"\n   ✅ Wrote {len(meta_df)} images → {produtos_dir}")
    print(f"   ✅ Wrote metadata.csv → {meta_path}")

    # Department distribution
    dist = meta_df["departamento"].value_counts()
    for d in DEPARTMENTS:
        print(f"      {d}: {dist.get(d, 0):,}")


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def write_dataset_readme(output_dir: Path) -> None:
    """Write a small README.txt inside the dataset folder."""
    readme = textwrap.dedent("""\
        desafio-novacommerce-v1
        =======================
        Dataset sintético para o Desafio de Redes Neurais e Deep Learning — FIAP.

        Conteúdo
        --------
        reviews.csv     40 000 reviews rotuladas (ok / atencao / remover)
        produtos/       15 000 imagens 224×224 RGB em 10 departamentos
        metadata.csv    Mapeamento produto_id ↔ imagem ↔ departamento

        Departamentos
        -------------
        roupas_femininas, roupas_masculinas, calcados, acessorios,
        bolsas_e_malas, esportes, beleza, joias_e_relogios, casa, infantil

        Fontes originais (Kaggle)
        -------------------------
        • olistbr/brazilian-ecommerce        (CC BY-NC-SA 4.0)
        • paramaggarwal/fashion-product-images-small  (MIT)

        Os dados foram transformados, amostrados e re-rotulados para fins
        educacionais. Parte das reviews são sintéticas, geradas a partir de
        templates, para garantir balanceamento e volume adequados.
    """)
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")


def create_zip(output_dir: Path) -> Path:
    """Create the final .zip archive."""
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    print(f"\n📦 Creating {zip_path} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(output_dir):
            for fname in tqdm(files, desc="   Compressing", leave=False):
                fpath = Path(root) / fname
                arcname = fpath.relative_to(output_dir.parent)
                zf.write(fpath, arcname)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Archive ready: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build desafio-novacommerce-v1 dataset from Kaggle sources."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("desafio-novacommerce-v1"),
        help="Output directory (default: ./desafio-novacommerce-v1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".kaggle_cache"),
        help="Directory to cache downloaded Kaggle datasets",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download (use cached data in --cache-dir)",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip .zip archive creation",
    )
    args = parser.parse_args()

    random.seed(SEED)
    output_dir: Path = args.output_dir.resolve()
    cache_dir: Path = args.cache_dir.resolve()

    print("=" * 60)
    print("  NovaCommerce Dataset Builder")
    print("=" * 60)
    print(f"  Output   : {output_dir}")
    print(f"  Cache    : {cache_dir}")
    print(f"  Reviews  : {REVIEWS_TARGET:,}")
    print(f"  Images   : {IMAGES_TARGET:,} ({IMAGES_PER_DEPT}/dept × {NUM_DEPARTMENTS})")
    print("=" * 60)

    # --- Download phase ---------------------------------------------------
    olist_dir = cache_dir / "olist"
    fashion_dir = cache_dir / "fashion"

    if not args.skip_download:
        _download_kaggle_dataset("olistbr/brazilian-ecommerce", olist_dir)
        _download_kaggle_dataset(
            "paramaggarwal/fashion-product-images-small", fashion_dir
        )
    else:
        print("\n⏭  Skipping downloads (--skip-download)")

    # --- Build phase ------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    build_reviews(olist_dir, output_dir)
    build_images_and_metadata(fashion_dir, output_dir)
    write_dataset_readme(output_dir)

    # --- Package ----------------------------------------------------------
    if not args.no_zip:
        create_zip(output_dir)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
