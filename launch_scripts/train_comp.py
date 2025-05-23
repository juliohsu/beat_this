import argparse
from pathlib import Path
import sys
import os
from datetime import datetime

# import os
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
# os.environ["TORCHINDUCTOR_DISABLE"] = "1"
import requests

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis
from beat_this.inference import load_checkpoint

from launch_scripts.compute_paper_metrics import plmodel_setup

# Simple class to tee output to both stdout/stderr and a file
class Tee:
    def __init__(self, original_stream, file):
        self.original_stream = original_stream
        self.file = file
        self.encoding = original_stream.encoding  # Add encoding attribute
        
    def write(self, message):
        self.original_stream.write(message)
        self.file.write(message)
        self.file.flush()  # Make sure log is written immediately
        
    def flush(self):
        self.original_stream.flush()
        self.file.flush()
        
    # Add any other attributes that might be accessed
    def __getattr__(self, name):
        return getattr(self.original_stream, name)
        
        
def freeze_model_component(model, component_name):
    """Freeze a specific component of the model by setting requires_grad=False for all parameters."""
    if hasattr(model.model, component_name):
        component = getattr(model.model, component_name)
        for param in component.parameters():
            param.requires_grad = False
        print(f"Freezing {component_name} component")
    else:
        print(f"Warning: Component {component_name} not found in model")

def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)

    print("Starting a new run with the following parameters:")
    print(args)

    params_str = f"{'noval ' if not args.val else ''}{'hung ' if args.hung_data else ''}{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}{args.loss}-h{args.transformer_dim}-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}{' nosumH ' if not args.sum_head else ''}{' nopartialT ' if not args.partial_transformers else ''}"
    if args.logger == "wandb":
        logger = WandbLogger(
            entity="juliohsu", project="beat_this", name=f"{args.name} {params_str}".strip()
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    # print('parent file', Path(__file__).parent.relative_to(Path.cwd()) / "data")
    data_dir = Path("/home/julio.hsu/beat_this/beat_this/data")
    print('data_dir', data_dir)
    # data_dir2 = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    checkpoint_dir = (
        Path(__file__).parent.relative_to(Path.cwd()) / "checkpoints"
    )
    print('checkpoint_dir', checkpoint_dir)
    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {
            "kind": "permute",
            "min_count": 1,
            "max_count": 6,
            "min_len": 0.1,
            "max_len": 2,
            "min_parts": 5,
            "max_parts": 9,
        }

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
    )
    datamodule.setup(stage="fit")

    # compute positive weights
    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    print("Using positive weights: ", pos_weights)
    dropout = {
        "frontend": args.frontend_dropout,
        "transformer": args.transformer_dropout,
    }

    # JOAQUIM
    # if args.checkpoint_path:
    #     print(f"Loading checkpoint from {args.checkpoint_path}")
    #     # Load from checkpoint for finetuning
    #     checkpoint = load_checkpoint(args.checkpoint_path)
    #     pl_model, _ = plmodel_setup(checkpoint, args.eval_trim_beats, args.dbn, args.gpu)
        
    #     # Update the learning rate for finetuning if specified
    #     if args.finetune_lr is not None:
    #         pl_model.hparams.lr = args.finetune_lr
    #         print(f"Setting finetuning learning rate to {args.finetune_lr}")
        
    #     # Freeze components if specified
    #     if args.freeze_frontend:
    #         freeze_model_component(pl_model, "frontend")
        
    #     if args.freeze_transformer:
    #         freeze_model_component(pl_model, "transformer_blocks")
            
    #     if args.freeze_heads:
    #         freeze_model_component(pl_model, "task_heads")
            
    #     # Print trainable parameters count
    #     total_params = sum(p.numel() for p in pl_model.parameters())
    #     trainable_params = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    #     print(f"Total parameters: {total_params}")
    #     print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%})")
    # else:
    #     # Create a new model from scratch
    #     pl_model = PLBeatThis(
    #         spect_dim=128,
    #         fps=50,
    #         transformer_dim=args.transformer_dim,
    #         ff_mult=4,
    #         n_layers=args.n_layers,
    #         stem_dim=32,
    #         dropout=dropout,
    #         lr=args.lr,
    #         weight_decay=args.weight_decay,
    #         pos_weights=pos_weights,
    #         head_dim=32,
    #         loss_type=args.loss,
    #         warmup_steps=args.warmup_steps,
    #         max_epochs=args.max_epochs,
    #         use_dbn=args.dbn,
    #         eval_trim_beats=args.eval_trim_beats,
    #         sum_head=args.sum_head,
    #         partial_transformers=args.partial_transformers,
    #     )

    pl_model = PLBeatThis(
        spect_dim=128,
        fps=50,
        transformer_dim=args.transformer_dim,
        ff_mult=4,
        n_layers=args.n_layers,
        stem_dim=32,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weights=pos_weights,
        head_dim=32,
        loss_type=args.loss,
        warmup_steps=args.warmup_steps,
        max_epochs=100,                                               #args.max_epochs,
        use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head,
        partial_transformers=args.partial_transformers,
    )

    # # # checkpoint inicio BIA
    # checkpoint_path = '/home/biabc/beatthis/repo_bt/beat_this/checkpoints/AAM_gtzan.ckpt'
    # checkpoint_path = 'https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt'
    # checkpoint = torch.load(local_path, map_location='cuda')
    # adicionei esses params mode e freeze_layers , apesar de ter esse mode vc so usa essa funcao para finetuning
    # pl_model,trainer = plmodel_setup(checkpoint, args.eval_trim_beats, args.dbn, args.gpu)
    # # #checkpoint fim 

    # FINETUNING JULIO
    # url = 'https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt'
    # local_path = 'final0.ckpt'
    # with open(local_path, 'wb') as f:
    #     f.write(requests.get(url).content)
    # print(f"Loading checkpoint from {local_path}")
    # # Load from checkpoint for finetuning
    # checkpoint = load_checkpoint(local_path)
    # pl_model, _ = plmodel_setup(checkpoint, args.eval_trim_beats, args.dbn, args.gpu)
    
    # Freeze components if specified
    # freeze_model_component(pl_model, "frontend")
    # freeze_model_component(pl_model, "transformer_blocks")
    # freeze_model_component(pl_model, "task_heads")
        
    # Print trainable parameters count
    total_params = sum(p.numel() for p in pl_model.parameters())
    trainable_params = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%})")

    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            raise ValueError("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    # save only the last model
    callbacks.append(
        ModelCheckpoint(
            every_n_epochs=1,
            dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip(),
        )
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpu,
        strategy="ddp" if len(args.gpu) > 1 else "auto",
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
    )

    trainer.fit(pl_model, datamodule)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, nargs='+', default=[0],
                    help="GPU IDs to use for training (e.g., --gpu 0 1 for using GPUs 0 and 1)")

    parser.add_argument(
        "--force-flash-attention", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--compile",
        action="store",
        nargs="*",
        type=str,
        default=["frontend", "transformer_blocks", "task_heads"],
        help="Which model parts to compile, among frontend, transformer_encoder, task_heads",
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument(
        "--frontend-dropout",
        type=float,
        default=0.1,
        help="dropout rate to apply in the frontend",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply in the main transformer blocks",
    )
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50, help="The spectrograms fps.")
    parser.add_argument(
        "--loss",
        type=str,
        default="shift_tolerant_weighted_bce",
        choices=[
            "shift_tolerant_weighted_bce",
            "fast_shift_tolerant_weighted_bce",
            "weighted_bce",
            "bce",
        ],
        help="The loss to use",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="warmup steps for optimizer"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="max epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="batch size for training"
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument(
        "--train-length",
        type=int,
        default=1500,
        help="maximum seq length for training in frames",
    )
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use madmom postprocessing DBN",
    )
    parser.add_argument(
        "--eval-trim-beats",
        metavar="SECONDS",
        type=float,
        default=5,
        help="Skip the first given seconds per piece in evaluating (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frequency",
        metavar="N",
        type=int,
        default=5,
        help="validate every N epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed tempo aumentation",
    )
    parser.add_argument(
        "--pitch-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed pitch aumentation",
    )
    parser.add_argument(
        "--mask-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use online mask aumentation",
    )
    parser.add_argument(
        "--sum-head",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use SumHead instead of two separate Linear heads",
    )
    parser.add_argument(
        "--partial-transformers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use Partial transformers in the frontend",
    )
    parser.add_argument(
        "--length-based-oversampling-factor",
        type=float,
        default=0.65,
        help="The factor to oversample the long pieces in the dataset. Set to 0 to only take one excerpt for each piece.",
    )
    parser.add_argument(
        "--val",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train on all data, including validation data, escluding test data. The validation metrics will still be computed, but they won't carry any meaning.",
    )
    parser.add_argument(
        "--hung-data",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Limit the training to Hung et al. data. The validation will still be computed on all datasets.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If given, the CV fold number to *not* train on (0-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generators.",
    )
    parser.add_argument(
        "--checkpoint-path", 
        type=str, 
        default=None,
        help="Path to checkpoint for finetuning. Can be a local path or a URL."
    )
    parser.add_argument(
        "--finetune-lr", 
        type=float, 
        default=None,
        help="Learning rate to use when finetuning from a checkpoint. If not specified, uses the learning rate from the checkpoint."
    )
    parser.add_argument(
        "--freeze-frontend",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Freeze the frontend components during finetuning",
    )
    parser.add_argument(
        "--freeze-transformer",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Freeze the transformer blocks during finetuning",
    )
    parser.add_argument(
        "--freeze-heads",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Freeze the task heads during finetuning",
    )
    parser.add_argument(
        "--save-logs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save terminal logs to a file in terminal_logs directory",
    )

    args = parser.parse_args()

    # Set up logging to file if requested
    if args.save_logs:
        # Create terminal_logs directory if it doesn't exist
        os.makedirs("terminal_logs", exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"terminal_logs/train_comp_{timestamp}.log"
        
        # Redirect stdout and stderr to the log file
        log_file = open(log_filename, 'w')
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        
        print(f"Logging output to {log_filename}")
    
    main(args)


