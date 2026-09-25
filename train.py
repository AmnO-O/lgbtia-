"""
Main Training Entrypoint for StereoQueerEval & Toxic Classification Pipelines
Supports CLI flags, JSON configuration files, and 2-phase fine-tuning.

Usage:
  # Train with mmBERT / ModernBERT backbone (2-phase training)
  python train.py --embed_source mmbert --model mmbert_transformer --epochs 20

  # Train scratch Transformer baseline
  python train.py --embed_source scratch --model transformer --epochs 30 --lr 0.0001

  # Train Bi-LSTM baseline
  python train.py --embed_source scratch --model bilstm --epochs 25
"""

import os
import sys
import json
import argparse
import torch

from pipeline.config import PipelineConfig
from pipeline.data import DataPipeline
from pipeline.trainer import StereoQueerTrainer
from pipeline.models.transformer import PytorchTransformerModel
from pipeline.models.lstm import PytorchRNNLSTM
from pipeline.models.rnn import VanillaRNNModel
from pipeline.models.mmbert import MMBertTransformerModel
from pipeline.models.classifier import FeatureClassifier
from pipeline.models.task_b_class_aware import TaskBClassAwareAttentionModel
from pipeline.task_b_data import TaskBRoleDataset
from pipeline.task_b_trainer import TaskBTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="StereoQueerEval Python Training Pipeline")
    
    # Task & Model
    parser.add_argument("--task", type=str, default="stereoqueer", choices=["stereoqueer", "toxic"],
                        help="Target task dataset to train")
    parser.add_argument("--target_task", type=str, default="all", choices=["all", "st", "hs", "tg"],
                        help="Train a specific subtask ('st'=Stereotype, 'hs'=Hate Speech, 'tg'=Target Identity) or 'all'")
    parser.add_argument("--embed_source", type=str, default="mmbert", choices=["mmbert", "scratch"],
                        help="Embedding source: pretrained mmBERT/ModernBERT or scratch embeddings")
    parser.add_argument("--model", type=str, default="mmbert_transformer",
                        choices=["mmbert_transformer", "task_b_class_aware", "feature_mlp", "transformer", "bilstm", "rnn"],
                        help="Model architecture ('task_b_class_aware' for Class-Aware Attention on Task B)")
    parser.add_argument("--use_query_interaction", action="store_true", default=True,
                        help="Enable Layer 2 Multi-Head Self-Attention interaction between label queries (Ablation H2)")
    parser.add_argument("--no_query_interaction", dest="use_query_interaction", action="store_false",
                        help="Disable Layer 2 MHSA for Ablation Study H2")
    parser.add_argument("--mmbert_model_name", type=str, default="jhu-clsp/mmbert-base",
                        help="Hugging Face model ID for ModernBERT / mmBERT")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (single-phase)")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for heads/encoder")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token/word sequence length")
    
    # 2-Phase settings
    parser.add_argument("--two_phase", action="store_true", default=True,
                        help="Enable 2-phase training (frozen phase then unfreeze last N layers)")
    parser.add_argument("--freeze_epochs", type=int, default=15, help="Max epochs for phase 1 (frozen backbone)")
    parser.add_argument("--unfreeze_epochs", type=int, default=15, help="Max epochs for phase 2 (unfrozen backbone)")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of last encoder blocks to unfreeze")
    parser.add_argument("--unfreeze_lr", type=float, default=2e-5, help="Learning rate for unfrozen backbone")
    
    # Directories
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset TSV/CSVs")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for saved models")
    parser.add_argument("--config_file", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--generate_sample_if_missing", action="store_true", default=True,
                        help="Auto-generate synthetic sample TSV files if data directory is empty")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        config = PipelineConfig.from_json(args.config_file)
    else:
        config = PipelineConfig(
            task=args.task,
            target_task=args.target_task,
            embed_source=args.embed_source,
            model_type=args.model,
            use_query_interaction=args.use_query_interaction,
            mmbert_model_name=args.mmbert_model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            max_length=args.max_length,
            two_phase=args.two_phase,
            freeze_phase_epochs=args.freeze_epochs,
            unfreeze_phase_epochs=args.unfreeze_epochs,
            unfreeze_layers=args.unfreeze_layers,
            unfreeze_lr=args.unfreeze_lr,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )

    print("==================================================")
    print("StereoQueerEval Python Training Pipeline")
    print(f"Task:         {config.task}")
    print(f"Target Subtask: {config.target_task} ({'Stereotype' if config.target_task == 'st' else 'Hate Speech' if config.target_task == 'hs' else 'Target ID' if config.target_task == 'tg' else 'Multi-task (All)'})")
    print(f"Embed Source: {config.embed_source}")
    print(f"Model Type:   {config.model_type}")
    print(f"Batch Size:   {config.batch_size}")
    print(f"Output Dir:   {config.output_dir}")
    print("==================================================")

    data_pipeline = DataPipeline(config)
    found_files = data_pipeline.find_data_files()
    
    if not found_files and args.generate_sample_if_missing:
        print(f"No TSV files detected in '{config.data_dir}'. Generating synthetic sample data...")
        from generate_sample_data import generate_sample_dataset
        generate_sample_dataset(data_dir=config.data_dir)
        found_files = data_pipeline.find_data_files()

    print(f"Loaded {len(found_files)} data file(s): {found_files}")
    data_pipeline.load_data(found_files)
    df_train, df_val = data_pipeline.split_data()
    print(f"Dataset Split (by video): Train={len(df_train)} rows | Validation={len(df_val)} rows")
    train_batches = (len(df_train) + config.batch_size - 1) // config.batch_size
    val_batches = (len(df_val) + config.batch_size - 1) // config.batch_size
    print(f"Batch Configuration: Batch Size={config.batch_size} -> {train_batches} train batches/epoch, {val_batches} val batches/epoch")

    # Save vocabulary if scratch
    tokenizer = None
    if config.embed_source == "mmbert":
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading pretrained tokenizer & model: {config.mmbert_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(config.mmbert_model_name)
        backbone = AutoModel.from_pretrained(config.mmbert_model_name)

        if config.model_type == "task_b_class_aware":
            print("Initializing Task B Class-Aware Multi-Head Attention Architecture...")
            from torch.utils.data import DataLoader
            train_ds = TaskBRoleDataset(df_train, tokenizer, max_len=config.max_length)
            val_ds = TaskBRoleDataset(df_val, tokenizer, max_len=config.max_length)
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

            model = TaskBClassAwareAttentionModel(
                mmbert_model=backbone,
                d_model=config.mmbert_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_query_interaction=config.use_query_interaction
            )
            is_task_b = True
            is_mmbert_tf = False
        elif config.model_type == "feature_mlp":
            train_loader, val_loader = data_pipeline.create_dataloaders(tokenizer=tokenizer)
            model = FeatureClassifier(in_dim=config.mmbert_dim)
            is_task_b = False
            is_mmbert_tf = False
        else:
            train_loader, val_loader = data_pipeline.create_dataloaders(tokenizer=tokenizer)
            model = MMBertTransformerModel(
                backbone,
                d_model=config.mmbert_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                target_dim=config.target_dim
            )
            is_task_b = False
            is_mmbert_tf = True
    else:
        is_task_b = False
        train_loader, val_loader = data_pipeline.create_dataloaders()
        vocab_path = os.path.join(config.output_dir, "stereoqueer_vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(data_pipeline.vocab, f)
        print(f"Saved vocabulary ({len(data_pipeline.vocab)} words) to {vocab_path}")

        vocab_size = len(data_pipeline.vocab)
        if config.model_type == "bilstm":
            model = PytorchRNNLSTM(
                vocab_size=vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                target_dim=config.target_dim
            )
        elif config.model_type == "rnn":
            model = VanillaRNNModel(
                vocab_size=vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                target_dim=config.target_dim
            )
        else:
            model = PytorchTransformerModel(
                vocab_size=vocab_size,
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                target_dim=config.target_dim
            )
        is_mmbert_tf = False

    # Save active pipeline config
    config_save_path = os.path.join(config.output_dir, "pipeline_config.json")
    config.save_json(config_save_path)
    print(f"Pipeline configuration saved to: {config_save_path}")

    # Launch Trainer
    if is_task_b:
        trainer = TaskBTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            df_val=df_val
        )
    else:
        trainer = StereoQueerTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            df_val=df_val,
            is_mmbert_tf=is_mmbert_tf
        )
    result = trainer.train()
    print("\nTraining completed successfully!")
    print(f"Best model checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
