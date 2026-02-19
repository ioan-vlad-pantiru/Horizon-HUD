"""
Convert trained Keras model to TensorFlow Lite format.
Supports INT8 quantization with representative dataset calibration.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.bdd100k_loader import BDD100KLoader


def representative_dataset_gen(
    dataset_loader: BDD100KLoader,
    num_samples: int = 100
) -> Generator:
    """
    Generate representative dataset for INT8 quantization calibration.
    
    Args:
        dataset_loader: Dataset loader instance
        num_samples: Number of samples to use for calibration
    
    Yields:
        Representative input samples
    """
    for i in range(min(num_samples, len(dataset_loader))):
        sample = dataset_loader[i]
        image = sample['image']
        
        # Convert to uint8 if needed
        if image.dtype == np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        yield [np.expand_dims(image_uint8, axis=0)]


def convert_to_tflite(
    model_path: str,
    output_path: str,
    quantization: str = "int8",
    representative_dataset: Generator = None,
    input_size: tuple = (320, 320)
):
    """
    Convert Keras model to TFLite format.
    
    Args:
        model_path: Path to saved Keras model
        output_path: Output path for TFLite model
        quantization: Quantization type ('int8', 'fp16', 'fp32')
        representative_dataset: Generator for representative dataset (for INT8)
        input_size: Input image size
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to concrete function
    concrete_func = model.__call__.get_concrete_function(
        tf.TensorSpec(shape=[1, *input_size, 3], dtype=tf.float32)
    )
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Apply quantization
    if quantization == "int8":
        if representative_dataset is None:
            raise ValueError("Representative dataset required for INT8 quantization")
        
        print("Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("INT8 quantization configured")
        
    elif quantization == "fp16":
        print("Applying FP16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("FP16 quantization configured")
    
    # Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    
    # Print model size
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras model to TFLite"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Keras model (.h5 or saved_model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for TFLite model"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "fp16", "fp32"],
        help="Quantization type"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="BDD100K dataset root (required for INT8 quantization)"
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default=None,
        help="Optional labels root path (for INT8 calibration dataset)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of samples for INT8 calibration"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        input_size = tuple(config['model']['input_size'])
        quantization = config['model']['quantization']
    else:
        input_size = (320, 320)
        quantization = args.quantization
    
    # Prepare representative dataset for INT8
    representative_dataset = None
    if quantization == "int8":
        if not args.dataset_root:
            raise ValueError("--dataset-root required for INT8 quantization")
        
        print("Loading representative dataset...")
        dataset_loader = BDD100KLoader(
            dataset_root=args.dataset_root,
            labels_root=args.labels_root,
            split="train",
            input_size=input_size,
        )
        
        representative_dataset = representative_dataset_gen(
            dataset_loader,
            num_samples=args.calibration_samples
        )
    
    # Convert
    convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        quantization=quantization,
        representative_dataset=representative_dataset,
        input_size=input_size,
    )


if __name__ == "__main__":
    main()
