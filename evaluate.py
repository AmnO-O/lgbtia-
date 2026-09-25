"""
Evaluation CLI script for StereoQueerEval models.
Loads a checkpoint and computes macro-F1, accuracy, exact-match target scores,
and per-language breakdown on validation or test sets.

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pt --config checkpoints/pipeline_config.json
"""

import os
import argparse
import pandas as pd
import torch

from pipeline.config import PipelineConfig
from pipeline.data import DataPipeline
from pipeline.metrics import evaluate_stereoqueer, print_metrics
from pipeline.models.transformer import PytorchTransformerModel
from pipeline.models.lstm import PytorchRNNLSTM
from pipeline.models.rnn import VanillaRNNModel
from pipeline.models.mmbert import MMBertTransformerModel
from pipeline.models.classifier import FeatureClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="StereoQueerEval Model Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="checkpoints/pipeline_config.json", help="Path to pipeline config JSON")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with data TSV files")
    return parser.parse_args()

def main():
    args = parse_args()
    if os.path.exists(args.config):
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig(data_dir=args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    data_pipeline = DataPipeline(config)
    data_pipeline.load_data()
    _, df_val = data_pipeline.split_data()

    if config.embed_source == "mmbert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(config.mmbert_model_name)
        _, val_loader = data_pipeline.create_dataloaders(tokenizer=tokenizer)
        backbone = AutoModel.from_pretrained(config.mmbert_model_name)
        model = MMBertTransformerModel(backbone, d_model=config.mmbert_dim)
        is_mmbert_tf = True
    else:
        _, val_loader = data_pipeline.create_dataloaders()
        model = PytorchTransformerModel(
            vocab_size=len(data_pipeline.vocab),
            embedding_dim=config.embedding_dim
        )
        is_mmbert_tf = False

    print(f"Loading weights from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    metrics, preds = evaluate_stereoqueer(model, df_val, val_loader, device, is_mmbert_tf=is_mmbert_tf)
    print_metrics(metrics, "VALIDATION EVALUATION RESULTS")

    if 'per_language' in preds:
        print("Per-Language Breakdown:")
        for lang, lmetrics in preds['per_language'].items():
            print(f"  [{lang}] Count: {lmetrics['count']} | ST F1: {lmetrics['st_f1']:.4f} | HS F1: {lmetrics['hs_f1']:.4f} | TG Exact: {lmetrics['tg_exact_match']:.4f}")

if __name__ == "__main__":
    main()
