"""
Inference CLI script for StereoQueerEval models.
Predicts stereotype presence, hate speech category, and target identity/scope
for a given YouTube comment, title, and description.

Usage:
  python predict.py --checkpoint checkpoints/best_model.pt \
                    --comment "All they care about is drama." \
                    --title "Pride Parade Highlights" \
                    --desc "Annual community celebration."
"""

import argparse
import json
from pipeline.inference import StereoQueerPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="StereoQueerEval Predictor")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="checkpoints/pipeline_config.json", help="Path to config JSON")
    parser.add_argument("--vocab", type=str, default="checkpoints/stereoqueer_vocab.json", help="Path to vocab JSON")
    parser.add_argument("--comment", type=str, required=True, help="YouTube comment text")
    parser.add_argument("--title", type=str, default="", help="YouTube video title")
    parser.add_argument("--desc", type=str, default="", help="YouTube video description")
    return parser.parse_args()

def main():
    args = parse_args()
    predictor = StereoQueerPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        vocab_path=args.vocab
    )
    result = predictor.predict_one(comment=args.comment, title=args.title, description=args.desc)
    print("\n--- Model Prediction Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
