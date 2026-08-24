from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "desafio_nps_fase_1.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"

TARGET_COLUMN = "is_detractor"
RANDOM_STATE = 42


EXPECTED_COLUMNS = {
    "customer_id",
    "customer_age",
    "customer_region",
    "customer_tenure_months",
    "order_id",
    "order_value",
    "items_quantity",
    "discount_value",
    "payment_installments",
    "delivery_time_days",
    "delivery_delay_days",
    "freight_value",
    "delivery_attempts",
    "customer_service_contacts",
    "resolution_time_days",
    "nps_score",
    "repeat_purchase_30d",
    "complaints_count",
    "csat_internal_score",
}


# Colunas que não podem entrar no treinamento.
EXCLUDED_FROM_MODEL = {
    "customer_id",
    "order_id",
    "nps_score",
    "nps_category",
    TARGET_COLUMN,
    "repeat_purchase_30d",
    "csat_internal_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa análise e modelagem do case NPS."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Caminho do CSV de entrada.",
    )

    return parser.parse_args()


def configure_output() -> None:
    """Cria as pastas de saída do projeto."""

    for directory in (
        PROCESSED_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )


def load_and_validate(path: Path) -> pd.DataFrame:
    """
    Carrega o CSV e valida o contrato básico dos dados.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {path}"
        )

    data = pd.read_csv(path)

    missing_columns = EXPECTED_COLUMNS.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError(
            "A base está vazia."
        )

    if data["customer_id"].duplicated().any():
        raise ValueError(
            "customer_id deveria ser único."
        )

    if data["order_id"].duplicated().any():
        raise ValueError(
            "order_id deveria ser único."
        )

    if not data["nps_score"].between(0, 10).all():
        raise ValueError(
            "nps_score deve estar entre 0 e 10."
        )

    return data


def prepare_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria target, categorias de NPS e faixas
    interpretáveis para a análise de negócio.
    """

    prepared = data.copy()

    prepared["nps_category"] = np.select(
        [
            prepared["nps_score"] <= 6,
            prepared["nps_score"] < 9,
        ],
        [
            "Detrator",
            "Neutro",
        ],
        default="Promotor",
    )

    prepared[TARGET_COLUMN] = (
        prepared["nps_score"] <= 6
    ).astype(int)

    prepared["delay_band"] = pd.cut(
        prepared["delivery_delay_days"],
        bins=[
            -1,
            0,
            2,
            4,
            np.inf,
        ],
        labels=[
            "Sem atraso",
            "1-2 dias",
            "3-4 dias",
            "5+ dias",
        ],
    )

    prepared["complaints_band"] = pd.cut(
        prepared["complaints_count"],
        bins=[
            -1,
            1,
            3,
            5,
            np.inf,
        ],
        labels=[
            "0-1",
            "2-3",
            "4-5",
            "6+",
        ],
    )

    prepared["contacts_band"] = pd.cut(
        prepared["customer_service_contacts"],
        bins=[
            -1,
            0,
            1,
            2,
            np.inf,
        ],
        labels=[
            "0",
            "1",
            "2",
            "3+",
        ],
    )

    prepared["resolution_band"] = pd.cut(
        prepared["resolution_time_days"],
        bins=[
            -1,
            2,
            5,
            8,
            np.inf,
        ],
        labels=[
            "0-2 dias",
            "3-5 dias",
            "6-8 dias",
            "9+ dias",
        ],
    )

    return prepared


def business_group_table(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Resume segmentos em métricas de negócio.
    """

    table = (
        data.groupby(
            column,
            observed=True,
        )
        .agg(
            pedidos=(
                "order_id",
                "size",
            ),
            nps_medio=(
                "nps_score",
                "mean",
            ),
            taxa_detratores=(
                TARGET_COLUMN,
                "mean",
            ),
            taxa_promotores=(
                "nps_category",
                lambda values: (
                    values == "Promotor"
                ).mean(),
            ),
        )
        .reset_index()
    )

    table["taxa_detratores"] *= 100
    table["taxa_promotores"] *= 100

    return table


def save_eda_tables(
    data: pd.DataFrame,
) -> dict:
    """
    Salva base preparada, qualidade,
    segmentações e resumo executivo.
    """

    data.to_csv(
        PROCESSED_DIR / "nps_prepared.csv",
        index=False,
    )

    quality = pd.DataFrame(
        {
            "tipo": data.dtypes.astype(str),
            "ausentes": data.isna().sum(),
            "valores_unicos": data.nunique(
                dropna=False
            ),
        }
    )

    quality.to_csv(
        REPORTS_DIR / "data_quality.csv",
        index_label="coluna",
    )

    segment_columns = [
        "delay_band",
        "complaints_band",
        "contacts_band",
        "resolution_band",
        "customer_region",
    ]

    for column in segment_columns:
        table = business_group_table(
            data,
            column,
        )

        table.to_csv(
            REPORTS_DIR
            / f"segment_{column}.csv",
            index=False,
        )

    counts = data[
        "nps_category"
    ].value_counts()

    promoters = int(
        counts.get(
            "Promotor",
            0,
        )
    )

    detractors = int(
        counts.get(
            "Detrator",
            0,
        )
    )

    neutrals = int(
        counts.get(
            "Neutro",
            0,
        )
    )

    n_rows = len(data)

    traditional_nps = (
        100
        * (promoters - detractors)
        / n_rows
    )

    summary = {
        "rows": n_rows,
        "columns_original": len(
            EXPECTED_COLUMNS
        ),
        "missing_values": int(
            data[
                list(EXPECTED_COLUMNS)
            ]
            .isna()
            .sum()
            .sum()
        ),
        "duplicate_rows": int(
            data[
                list(EXPECTED_COLUMNS)
            ]
            .duplicated()
            .sum()
        ),
        "mean_nps_score": round(
            float(
                data[
                    "nps_score"
                ].mean()
            ),
            3,
        ),
        "traditional_nps": round(
            traditional_nps,
            1,
        ),
        "detractor_rate_pct": round(
            100
            * detractors
            / n_rows,
            1,
        ),
        "passive_rate_pct": round(
            100
            * neutrals
            / n_rows,
            1,
        ),
        "promoter_rate_pct": round(
            100
            * promoters
            / n_rows,
            1,
        ),
    }

    output = (
        REPORTS_DIR
        / "business_summary.json"
    )

    output.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


def plot_eda(
    data: pd.DataFrame,
) -> None:
    """Gera gráficos da EDA."""

    order = [
        "Detrator",
        "Neutro",
        "Promotor",
    ]

    percentages = (
        data["nps_category"]
        .value_counts(
            normalize=True
        )
        .reindex(order)
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        x=percentages.index,
        y=percentages.values,
        ax=ax,
    )

    ax.set_title(
        "Distribuição das categorias de NPS"
    )

    ax.set_xlabel(
        "Categoria"
    )

    ax.set_ylabel(
        "Pedidos (%)"
    )

    for index, value in enumerate(
        percentages
    ):
        ax.text(
            index,
            value + 1,
            f"{value:.1f}%",
            ha="center",
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "01_nps_categories.png",
        dpi=180,
    )

    plt.close(fig)

    delay = business_group_table(
        data,
        "delay_band",
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.barplot(
        data=delay,
        x="delay_band",
        y="nps_medio",
        ax=ax,
    )

    ax.set_title(
        "NPS médio por faixa de atraso"
    )

    ax.set_xlabel(
        "Faixa de atraso"
    )

    ax.set_ylabel(
        "NPS médio"
    )

    ax.set_ylim(
        0,
        10,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "02_nps_by_delay.png",
        dpi=180,
    )

    plt.close(fig)

    scenarios = [
        (
            "Sem atraso",
            data[
                "delivery_delay_days"
            ]
            == 0,
        ),
        (
            "5+ dias de atraso",
            data[
                "delivery_delay_days"
            ]
            >= 5,
        ),
        (
            "0-1 reclamação",
            data[
                "complaints_count"
            ]
            <= 1,
        ),
        (
            "6+ reclamações",
            data[
                "complaints_count"
            ]
            >= 6,
        ),
        (
            "Sem contato",
            data[
                "customer_service_contacts"
            ]
            == 0,
        ),
        (
            "3+ contatos",
            data[
                "customer_service_contacts"
            ]
            >= 3,
        ),
    ]

    rates = pd.DataFrame(
        {
            "cenario": [
                name
                for name, _
                in scenarios
            ],
            "taxa_detratores": [
                100
                * data.loc[
                    mask,
                    TARGET_COLUMN,
                ].mean()
                for _, mask
                in scenarios
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=rates,
        y="cenario",
        x="taxa_detratores",
        ax=ax,
    )

    ax.set_title(
        "Taxa de detratores por cenário operacional"
    )

    ax.set_xlabel(
        "Detratores (%)"
    )

    ax.set_ylabel("")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "03_detractor_rate_scenarios.png",
        dpi=180,
    )

    plt.close(fig)

    correlation_columns = [
        "nps_score",
        "delivery_delay_days",
        "complaints_count",
        "customer_service_contacts",
        "resolution_time_days",
        "delivery_time_days",
        "freight_value",
        "order_value",
    ]

    correlation = data[
        correlation_columns
    ].corr(
        method="spearman"
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="vlag",
        ax=ax,
    )

    ax.set_title(
        "Associações de Spearman"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "04_correlation_heatmap.png",
        dpi=180,
    )

    plt.close(fig)


def make_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    """
    Padroniza numéricas e codifica categóricas.
    """

    categorical = (
        features.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numeric = (
        features.columns
        .difference(
            categorical
        )
        .tolist()
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical,
            ),
        ]
    )


def classification_metrics(
    name: str,
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict:
    """Calcula métricas da classificação."""

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "model": name,
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "average_precision":
            average_precision_score(
                y_true,
                probability,
            ),
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "precision_detractor":
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            ),
        "recall_detractor":
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            ),
        "f1_detractor":
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            ),
    }


def train_and_evaluate(
    data: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Compara baseline, regressão logística
    e Random Forest.
    """

    model_data = data.drop(
        columns=[
            "delay_band",
            "complaints_band",
            "contacts_band",
            "resolution_band",
        ]
    )

    feature_columns = [
        column
        for column
        in model_data.columns
        if column
        not in EXCLUDED_FROM_MODEL
    ]

    features = model_data[
        feature_columns
    ]

    target = model_data[
        TARGET_COLUMN
    ]

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    models = {
        "Baseline majoritario":
            DummyClassifier(
                strategy="prior"
            ),

        "Regressao logistica":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Random forest":
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fitted = {}
    probabilities = {}

    cv_rows = []
    metric_rows = []

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(x_train),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        # A escolha do modelo acontece somente
        # dentro dos dados de treinamento.
        cv_result = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
            },
            return_train_score=True,
            n_jobs=1,
        )

        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_result[
                    "test_roc_auc"
                ].mean(),
                "cv_roc_auc_std": cv_result[
                    "test_roc_auc"
                ].std(),
                "cv_average_precision_mean": cv_result[
                    "test_average_precision"
                ].mean(),
                "cv_average_precision_std": cv_result[
                    "test_average_precision"
                ].std(),
                "train_roc_auc_mean": cv_result[
                    "train_roc_auc"
                ].mean(),
            }
        )

        # Após a validação cruzada, treinamos cada candidato com todo o conjunto de treino.
        pipeline.fit(
            x_train,
            y_train,
        )

        probability = pipeline.predict_proba(
            x_test
        )[:, 1]

        fitted[name] = pipeline
        probabilities[name] = probability

        metric_rows.append(
            classification_metrics(
                name,
                y_test,
                probability,
            )
        )

    # Resultados usados para escolher o modelo.
    cv_metrics = (
        pd.DataFrame(cv_rows)
        .sort_values(
            "cv_roc_auc_mean",
            ascending=False,
        )
    )

    cv_metrics.to_csv(
        REPORTS_DIR / "model_cv_metrics.csv",
        index=False,
    )

    # Holdout é mantido separado e serve
    # como avaliação final.
    metrics = (
        pd.DataFrame(metric_rows)
        .sort_values(
            "roc_auc",
            ascending=False,
        )
    )

    metrics.to_csv(
        REPORTS_DIR / "model_metrics.csv",
        index=False,
    )

    # Baseline não pode ser selecionado
    # como modelo final.
    eligible_cv = cv_metrics[
        cv_metrics["model"]
        != "Baseline majoritario"
    ]

    best_name = str(
        eligible_cv.iloc[0]["model"]
    )

    best_model = fitted[
        best_name
    ]

    best_probability = probabilities[
        best_name
    ]

    best_prediction = (
        best_probability >= 0.5
    ).astype(int)

    selected_cv = cv_metrics.loc[
        cv_metrics["model"] == best_name
    ].iloc[0]

    details = {
        "chosen_model":
            best_name,
            
        "model_selection": (
            "Maior ROC-AUC media em "
            "StratifiedKFold de 5 folds "
            "aplicado somente ao conjunto de treino"
        ),

        "cv_roc_auc_mean": float(
            selected_cv["cv_roc_auc_mean"]
        ),

        "cv_roc_auc_std": float(
            selected_cv["cv_roc_auc_std"]
        ),

        "target_definition":
            (
                "1 quando nps_score <= 6; "
                "0 quando nps_score > 6"
            ),

        "excluded_from_model":
            sorted(
                EXCLUDED_FROM_MODEL
            ),

        "train_rows":
            len(x_train),

        "test_rows":
            len(x_test),

        "test_detractor_rate":
            float(
                y_test.mean()
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                best_prediction,
            ).tolist(),

        "classification_report":
            classification_report(
                y_test,
                best_prediction,
                output_dict=True,
                zero_division=0,
            ),
    }

    (
        REPORTS_DIR
        / "model_details.json"
    ).write_text(
        json.dumps(
            details,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    joblib.dump(
        best_model,
        MODELS_DIR
        / "detractor_classifier.joblib",
    )

    scored_test = x_test.copy()

    scored_test[
        "actual_is_detractor"
    ] = y_test

    scored_test[
        "predicted_detractor_probability"
    ] = best_probability

    scored_test.to_csv(
        PROCESSED_DIR
        / "test_predictions.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for name, probability in (
        probabilities.items()
    ):

        fpr, tpr, _ = roc_curve(
            y_test,
            probability,
        )

        auc = roc_auc_score(
            y_test,
            probability,
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{name} "
                f"(AUC={auc:.3f})"
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_title(
        "Curva ROC dos modelos"
    )

    ax.set_xlabel(
        "Taxa de falso positivo"
    )

    ax.set_ylabel(
        "Taxa de verdadeiro positivo"
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "05_model_roc_curve.png",
        dpi=180,
    )

    plt.close(fig)

    importance = permutation_importance(
        best_model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_table = (
        pd.DataFrame(
            {
                "feature":
                    x_test.columns,

                "importance_mean":
                    importance.importances_mean,

                "importance_std":
                    importance.importances_std,
            }
        )
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_table.to_csv(
        REPORTS_DIR
        / "model_feature_importance.csv",
        index=False,
    )

    top = (
        importance_table
        .head(10)
        .sort_values(
            "importance_mean"
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )

    ax.set_title(
        f"Importância das variáveis — {best_name}"
    )

    ax.set_xlabel(
        "Queda média da ROC-AUC"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "06_model_feature_importance.png",
        dpi=180,
    )

    plt.close(fig)

    return (
        best_model,
        metrics,
    )


def main() -> None:

    args = parse_args()

    configure_output()

    raw_data = load_and_validate(
        args.input
    )

    prepared_data = prepare_data(
        raw_data
    )

    summary = save_eda_tables(
        prepared_data
    )

    plot_eda(
        prepared_data
    )

    _, metrics = train_and_evaluate(
        prepared_data
    )

    print(
        "\nAnálise concluída com sucesso."
    )

    print(
        f"Pedidos analisados: "
        f"{summary['rows']}"
    )

    print(
        f"NPS tradicional: "
        f"{summary['traditional_nps']}"
    )

    print(
        "\nMétricas dos modelos:"
    )

    print(
        metrics
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        "\nResultados salvos em:"
    )

    print(
        REPORTS_DIR
    )


if __name__ == "__main__":
    main()