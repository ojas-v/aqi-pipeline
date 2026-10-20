"""Stage 1 - Data ingestion with defensive loading and schema validation.

Why a dedicated stage: downstream code should be able to *assume* a valid,
correctly-typed DataFrame. All "is the file there / is it the right shape"
logic lives here and nowhere else (single responsibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import EXPECTED_SCHEMA, TIMESTAMP_COL
from src.exceptions import DataIngestionError, SchemaValidationError
from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestor:
    """Loads a CSV and enforces the expected schema.

    Parameters
    ----------
    path:
        Location of the raw CSV file.
    schema:
        Mapping ``column -> kind`` where kind is ``datetime``, ``numeric`` or
        ``category``. Defaults to :data:`src.config.EXPECTED_SCHEMA`.
    """

    def __init__(self, path: Path, schema: Mapping[str, str] = EXPECTED_SCHEMA) -> None:
        self._path = Path(path)
        self._schema = schema

    def load(self) -> pd.DataFrame:
        """Read, validate and type-coerce the dataset (the only public method)."""
        df = self._read_csv()
        self._validate_schema(df)
        df = self._coerce_types(df)
        logger.info("Ingested %d rows x %d columns from %s", *df.shape, self._path.name)
        return df

    # ------------------------------------------------------------------ #
    def _read_csv(self) -> pd.DataFrame:
        """Read the file, translating low-level errors into domain errors."""
        try:
            df = pd.read_csv(self._path)
        except FileNotFoundError as exc:
            raise DataIngestionError(f"Raw data file not found: {self._path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise DataIngestionError(f"Raw data file is empty: {self._path}") from exc
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DataIngestionError(f"Malformed CSV ({self._path}): {exc}") from exc
        if df.empty:
            raise DataIngestionError("CSV parsed correctly but contains zero rows.")
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Fail fast if required columns are missing; only warn on extras."""
        missing = [c for c in self._schema if c not in df.columns]
        if missing:
            raise SchemaValidationError(f"Missing required columns: {missing}")
        extra = [c for c in df.columns if c not in self._schema]
        if extra:
            logger.warning("Ignoring %d unexpected column(s): %s", len(extra), extra)
        df.drop(columns=extra, inplace=True)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to their logical dtype.

        ``errors='coerce'`` converts unparseable cells (e.g. the string "N/A")
        to NaN/NaT instead of crashing; we log how many were affected so
        nothing is silently lost.
        """
        for col, kind in self._schema.items():
            if kind == "numeric":
                before = df[col].notna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                lost = int(before - df[col].notna().sum())
                if lost:
                    logger.warning("Column '%s': %d non-numeric cell(s) coerced to NaN", col, lost)
            elif kind == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif kind == "category":
                # Plain object dtype with NaN for gaps: the most portable
                # representation for sklearn imputers/encoders across pandas versions.
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

        bad_ts = int(df[TIMESTAMP_COL].isna().sum())
        if bad_ts:
            logger.warning("Dropping %d row(s) with unparseable timestamps", bad_ts)
            df = df.dropna(subset=[TIMESTAMP_COL])
        # Chronological order: required for any future time-aware validation.
        return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
