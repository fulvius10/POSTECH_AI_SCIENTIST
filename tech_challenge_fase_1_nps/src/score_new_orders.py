from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    PROJECT_DIR
    / "models"
    / "detractor_classifier.joblib"
)

DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "scored_orders.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica o modelo treinado a novos pedidos."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV com os pedidos que serão pontuados.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Caminho do modelo treinado.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Caminho do CSV de saída.",
    )

    return parser.parse_args()


def risk_band(probability: pd.Series) -> pd.Categorical:
    """Converte a probabilidade prevista em faixa operacional de risco."""

    return pd.cut(
        probability,
        bins=[-0.01, 0.50, 0.75, 1.00],
        labels=["Baixo", "Alto", "Critico"],
    )


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {args.model}"
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {args.input}"
        )

    model = joblib.load(args.model)

    data = pd.read_csv(args.input)

    required_features = list(
        model.feature_names_in_
    )

    missing_features = sorted(
        set(required_features).difference(
            data.columns
        )
    )

    if missing_features:
        raise ValueError(
            "Colunas necessárias ausentes: "
            f"{missing_features}"
        )

    # result = data.copy()
    
    id_columns = [
        column
        for column in ["customer_id", "order_id"]
        if column in data.columns
    ]

    output_columns = list(
        dict.fromkeys(
            id_columns + required_features
        )
    )

    result = data[output_columns].copy()

    result["detractor_probability"] = (
        model.predict_proba(
            data[required_features]
        )[:, 1]
    )

    result["risk_band"] = risk_band(
        result["detractor_probability"]
    )

    result = result.sort_values(
        "detractor_probability",
        ascending=False,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    print(
        f"{len(result)} pedidos pontuados."
    )

    print(
        f"Saída: {args.output}"
    )


if __name__ == "__main__":
    main()