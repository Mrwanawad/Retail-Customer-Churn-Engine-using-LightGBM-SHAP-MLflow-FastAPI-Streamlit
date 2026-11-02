"""Fetch the Dunnhumby Complete Journey CSVs from Kaggle into data/raw/dunnhumby (causal_data is skipped: 700 MB, unused)."""

from __future__ import annotations

import shutil
from pathlib import Path

from churn.config import get_logger, load_config

log = get_logger(__name__)
FILES = ["campaign_desc", "campaign_table", "coupon", "coupon_redempt", "hh_demographic", "product", "transaction_data"]


def main() -> None:
    import kagglehub

    src = Path(kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey"))
    dst = load_config().path("dunnhumby")
    for f in FILES:
        shutil.copy(src / f"{f}.csv", dst / f"{f}.csv")
    log.info("copied %d files -> %s", len(FILES), dst)


if __name__ == "__main__":
    main()
