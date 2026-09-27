"""Download the pretrained backbones once and store them as safetensors (needs network).

Usage::

    python -m dxa_qc.training.backbones --weights-dir weights/backbones

Inference never downloads anything: the container ships these files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dxa_qc.models.quality.embeddings import ALL_SPECS, export_backbones


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="training config: export its backbones too")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    specs = {s.key: s for s in ALL_SPECS.values()}
    if args.config:
        from dxa_qc.training.config import load_training_config

        specs.update({s.key: s for s in load_training_config(args.config).all_backbones()})
    hashes = export_backbones(args.weights_dir, list(specs.values()))
    (args.weights_dir / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    logging.info("backbones: %s", hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
