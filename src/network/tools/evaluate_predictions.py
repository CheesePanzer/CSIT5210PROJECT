"""Evaluate a saved prediction CSV without retraining a model."""

import argparse
import json
from pathlib import Path

from src.network.evaluation import evaluate_prediction_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_prediction_csv(args.predictions, args.num_classes)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
