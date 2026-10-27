"""Primary entry point: real-data PM2.5 forecasting; synthetic pipeline is optional."""
from __future__ import annotations
import argparse
import sys
from forecasting import main as run_forecast

def run_synthetic() -> int:
    from data_generator import AQIDataGenerator
    from src.cleaning import DataCleaner, DistributionReporter
    from src.config import NUMERIC_FEATURES, PipelineConfig
    from src.exceptions import PipelineError
    from src.ingestion import DataIngestor
    from src.logger import configure_logging, get_logger
    from src.modeling import ModelEvaluator, ModelTrainer, split_features_target
    logger = get_logger("main")
    configure_logging()
    try:
        config = PipelineConfig()
        logger.info("Synthetic robustness mode: generate -> ingest -> clean -> train -> evaluate")
        generator = AQIDataGenerator(config)
        generator.save(generator.generate())
        raw = DataIngestor(config.raw_data_path).load()
        ruled = DataCleaner().clean(raw)
        X_train, X_test, y_train, y_test = split_features_target(ruled, config)
        raw_train = raw.loc[X_train.index]
        DistributionReporter.report_cleaning_effect(raw_train=raw_train, rule_clean_train=X_train,
            numeric_cols=NUMERIC_FEATURES, strategy=config.impute_strategy, iqr_multiplier=config.iqr_multiplier)
        trainer = ModelTrainer(config)
        cv_table = trainer.cross_validate_candidates(X_train, y_train)
        best_name, model = trainer.fit_best(cv_table, X_train, y_train)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test, y_test)
        evaluator.print_report(cv_table, best_name, metrics, evaluator.feature_importances(model))
        evaluator.save(model, metrics, config, best_name)
    except PipelineError as exc:
        logger.error("Synthetic pipeline aborted: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected synthetic pipeline failure")
        return 2
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="run the original synthetic AQI nowcast as a robustness demonstration")
    args = parser.parse_args()
    if args.synthetic:
        sys.exit(run_synthetic())
    run_forecast()
