"""Phase 3 - Orchestrator: Generation -> Ingestion -> Cleaning -> Training -> Evaluation.

Run:  python main.py
"""

from __future__ import annotations

import sys

from data_generator import AQIDataGenerator
from src.cleaning import DataCleaner, DistributionReporter
from src.config import NUMERIC_FEATURES, PipelineConfig
from src.exceptions import PipelineError
from src.ingestion import DataIngestor
from src.logger import configure_logging, get_logger
from src.modeling import ModelEvaluator, ModelTrainer, split_features_target

logger = get_logger("main")


def run(config: PipelineConfig) -> None:
    """Execute the five stages in order; each stage is an independent, testable class."""
    # ---- Stage 1: data generation -------------------------------------- #
    logger.info("STAGE 1/5  Data generation")
    generator = AQIDataGenerator(config)
    generator.save(generator.generate())

    # ---- Stage 2: ingestion -------------------------------------------- #
    logger.info("STAGE 2/5  Ingestion + schema validation")
    raw = DataIngestor(config.raw_data_path).load()

    # ---- Stage 3: cleaning --------------------------------------------- #
    logger.info("STAGE 3/5  Cleaning (stateless rules) + statistics report")
    ruled = DataCleaner().clean(raw)
    X_train, X_test, y_train, y_test = split_features_target(ruled, config)
    # 'before' = raw ingested values of the very same training rows
    # (DataCleaner preserves the index, so .loc gives an exact match).
    raw_train = raw.loc[X_train.index]
    DistributionReporter.report_cleaning_effect(
        raw_train=raw_train, rule_clean_train=X_train, numeric_cols=NUMERIC_FEATURES,
        strategy=config.impute_strategy, iqr_multiplier=config.iqr_multiplier)

    # ---- Stage 4: training --------------------------------------------- #
    logger.info("STAGE 4/5  Cross-validated model selection + final fit")
    trainer = ModelTrainer(config)
    cv_table = trainer.cross_validate_candidates(X_train, y_train)
    best_name, model = trainer.fit_best(cv_table, X_train, y_train)

    # ---- Stage 5: evaluation ------------------------------------------- #
    logger.info("STAGE 5/5  Hold-out evaluation")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)
    evaluator.print_report(cv_table, best_name, metrics, evaluator.feature_importances(model))
    evaluator.save(model, metrics, config, best_name)


def main() -> int:
    configure_logging()
    try:
        run(PipelineConfig())
    except PipelineError as exc:
        logger.error("Pipeline aborted: %s", exc)
        return 1
    except Exception:  # last-resort guard so the traceback is logged, not lost
        logger.exception("Unexpected failure")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
