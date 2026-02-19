"""
Training script for Horizon-HUD General Object Detection.
Trains SSD MobileNetV2 on BDD100K dataset.
"""

import sys
import yaml
import argparse
import math
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import mixed_precision

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.data.bdd100k_loader import BDD100KLoader
from ml.training.model_builder import build_ssd_mobilenet_v2
from ml.utils.class_mapping import NUM_CLASSES


def configure_acceleration():
    """Configure runtime acceleration (Metal GPU + mixed precision)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        print(f"Acceleration: GPU detected ({len(gpus)} device(s)), mixed precision enabled (mixed_float16)")
    else:
        mixed_precision.set_global_policy("float32")
        print("Acceleration: no GPU detected, using float32 on CPU")

class EpochCheckpoint(tf.keras.callbacks.Callback):
    """Save model weights every N epochs."""

    def __init__(self, checkpoint_dir: Path, every_n_epochs: int = 5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every_n_epochs = max(1, int(every_n_epochs))

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num % self.every_n_epochs == 0:
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch_num:03d}.weights.h5"
            self.model.save_weights(str(ckpt_path))
            print(f"\nSaved periodic checkpoint: {ckpt_path}")


class EpochStatsLogger(tf.keras.callbacks.Callback):
    """Print extra epoch stats that help training monitoring."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f" - lr: {lr:.6g}")


class EpochSampleMetrics(tf.keras.callbacks.Callback):
    """Compute simple box/class metrics on one fixed batch each epoch."""

    def __init__(self, sample_batch, box_output_name: str, class_output_name: str):
        super().__init__()
        self.sample_x, self.sample_y, self.sample_sw = sample_batch
        self.box_output_name = box_output_name
        self.class_output_name = class_output_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model(self.sample_x, training=False)
        pred_boxes = y_pred[self.box_output_name]
        pred_classes = y_pred[self.class_output_name]
        true_boxes = tf.cast(self.sample_y[self.box_output_name], pred_boxes.dtype)
        true_classes = tf.cast(self.sample_y[self.class_output_name], tf.int32)
        class_mask = tf.cast(self.sample_sw[self.class_output_name], tf.float32)

        # box MAE over valid anchors
        box_abs = tf.abs(pred_boxes - true_boxes)
        box_mask = tf.expand_dims(tf.cast(class_mask, pred_boxes.dtype), axis=-1)
        box_mae = tf.reduce_sum(box_abs * box_mask) / tf.maximum(tf.reduce_sum(box_mask), 1.0)

        # class accuracy over valid anchors
        pred_ids = tf.argmax(pred_classes, axis=-1, output_type=tf.int32)
        class_correct = tf.cast(tf.equal(pred_ids, true_classes), tf.float32) * class_mask
        cls_acc = tf.reduce_sum(class_correct) / tf.maximum(tf.reduce_sum(class_mask), 1.0)

        logs["sample_box_mae"] = float(box_mae.numpy())
        logs["sample_cls_acc"] = float(cls_acc.numpy())
        print(f" - sample_box_mae: {logs['sample_box_mae']:.6f} - sample_cls_acc: {logs['sample_cls_acc']:.6f}")


def load_config(config_path: str):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_callbacks(
    checkpoint_dir: Path,
    log_dir: Path,
    monitor: str = "val_loss",
    patience: int = 10,
    checkpoint_every_epochs: int = 5,
):
    """Create training callbacks."""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        EpochCheckpoint(checkpoint_dir=checkpoint_dir, every_n_epochs=checkpoint_every_epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        EpochStatsLogger(),
    ]
    return callbacks


def prepare_datasets(
    config: dict,
    dataset_root: str,
    labels_root: str = None,
    max_boxes: int = 600,
    box_output_name: str = "boxes",
    class_output_name: str = "classes",
    max_train_samples: int = None,
    max_val_samples: int = None,
):
    """Prepare train/val datasets."""
    input_size = tuple(config['model']['input_size'])
    
    train_loader = BDD100KLoader(
        dataset_root=dataset_root,
        labels_root=labels_root,
        split="train",
        input_size=input_size,
        preserve_diversity=config['dataset']['preserve_diversity']['day_night'],
        max_samples=max_train_samples,
    )
    
    val_loader = BDD100KLoader(
        dataset_root=dataset_root,
        labels_root=labels_root,
        split="val",
        input_size=input_size,
        preserve_diversity=False,
        max_samples=max_val_samples,
    )
    
    batch_size = config['training']['batch_size']
    
    train_dataset = train_loader.create_tf_dataset(
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        max_boxes=max_boxes,
    )
    
    val_dataset = val_loader.create_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        max_boxes=max_boxes,
    )

    def to_model_inputs(batch):
        x = batch["image"]
        class_mask = tf.cast(batch["classes"] >= 0, tf.float32)
        safe_classes = tf.where(batch["classes"] < 0, tf.zeros_like(batch["classes"]), batch["classes"])
        y = {
            box_output_name: batch["boxes"],
            class_output_name: safe_classes,
        }
        sample_weight = {
            box_output_name: tf.expand_dims(class_mask, axis=-1),
            class_output_name: class_mask,
        }
        return x, y, sample_weight

    train_dataset = train_dataset.map(to_model_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(to_model_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.repeat()
    
    return train_dataset, val_dataset, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Horizon-HUD object detector")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to BDD100K dataset root",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default=None,
        help="Optional path to labels folder if different from dataset root",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train samples for quick sanity runs",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on val samples for quick sanity runs",
    )
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=None,
        help="Optional fixed train/val steps for fast preflight checks",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of epochs",
    )
    
    args = parser.parse_args()

    configure_acceleration()
    
    # Load configs
    exp_config = load_config(args.config)
    model_config_path = project_root / "ml" / "config" / "model_config.yaml"
    model_config = load_config(str(model_config_path))
    
    # Create output directories
    exp_name = exp_config['experiment']['name']
    
    output_dir = Path(exp_config['output']['model_dir'].format(experiment_name=exp_name))
    checkpoint_dir = Path(exp_config['output']['checkpoint_dir'].format(experiment_name=exp_name))
    log_dir = Path(exp_config['output']['log_dir'].format(experiment_name=exp_name))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs
    with open(output_dir / "experiment_config.yaml", 'w') as f:
        yaml.dump(exp_config, f)
    with open(output_dir / "model_config.yaml", 'w') as f:
        yaml.dump(model_config, f)
    
    # Build model
    print("Building model...")
    input_size = tuple(model_config['model']['input_size'])
    model = build_ssd_mobilenet_v2(
        input_shape=(*input_size, 3),
        num_classes=NUM_CLASSES,
        pretrained=exp_config['training']['pretrained'],
    )
    output_names = list(model.output_names)
    if isinstance(model.output, dict):
        output_keys = list(model.output.keys())
    else:
        output_keys = output_names
    if len(output_keys) != 2:
        raise ValueError(f"Expected 2 model outputs, got {len(output_keys)}: {output_keys}")
    box_output_name, class_output_name = output_keys[0], output_keys[1]

    model_anchors = int(model.output["boxes"].shape[1])

    # Prepare datasets
    print("Loading datasets...")
    train_dataset, val_dataset, train_loader, val_loader = prepare_datasets(
        model_config,
        args.dataset_root,
        args.labels_root,
        model_anchors,
        box_output_name,
        class_output_name,
        args.max_train_samples,
        args.max_val_samples,
    )
    
    print(f"Train samples: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    preview_x, preview_y, preview_sw = next(iter(train_dataset.take(1)))
    batch_size = exp_config['training']['batch_size']
    steps_per_epoch = math.ceil(len(train_loader) / batch_size)
    validation_steps = math.ceil(len(val_loader) / batch_size)
    if args.sanity_steps is not None:
        steps_per_epoch = max(1, int(args.sanity_steps))
        validation_steps = max(1, int(args.sanity_steps))
    epochs_to_run = args.epochs if args.epochs is not None else exp_config['training']['epochs']
    if args.resume:
        resume_path = args.resume
    else:
        resume_path = exp_config.get("training", {}).get("resume_from")

    if resume_path:
        if str(resume_path).endswith(".tflite"):
            raise ValueError(
                "Cannot resume training from a .tflite file. "
                "Use a Keras checkpoint (.keras/.weights.h5/.h5) or SavedModel directory."
            )
        print(f"Loading checkpoint: {resume_path}")
        if Path(resume_path).is_dir():
            prior_model = tf.keras.models.load_model(resume_path, compile=False)
            model.set_weights(prior_model.get_weights())
        elif str(resume_path).endswith(".keras"):
            prior_model = tf.keras.models.load_model(resume_path, compile=False)
            model.set_weights(prior_model.get_weights())
        else:
            model.load_weights(resume_path)
    
    # Compile model (per-output losses with masking via sample_weight)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=model_config['training']['learning_rate']
        ),
        loss={
            box_output_name: tf.keras.losses.Huber(),
            class_output_name: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        },
        loss_weights={
            box_output_name: model_config['training']['loss']['localization_weight'],
            class_output_name: model_config['training']['loss']['classification_weight'],
        },
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir,
        log_dir,
        monitor=exp_config['training']['early_stopping']['monitor'],
        patience=exp_config['training']['early_stopping']['patience'],
        checkpoint_every_epochs=exp_config['training']['checkpoint']['save_frequency'],
    )
    callbacks.append(EpochSampleMetrics((preview_x, preview_y, preview_sw), box_output_name, class_output_name))

    # Train
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_to_run,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save final model
    final_model_path = output_dir / "final_model.keras"
    model.save(str(final_model_path))
    print(f"Training complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
