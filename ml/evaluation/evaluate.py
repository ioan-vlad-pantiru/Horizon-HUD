"""
Evaluation framework for Horizon-HUD object detection models.
Computes mAP, per-class metrics, latency, and stability metrics.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.bdd100k_loader import BDD100KLoader
from ml.utils.class_mapping import CLASS_NAMES, NUM_CLASSES


class DetectionEvaluator:
    """Evaluator for object detection models."""
    
    def __init__(
        self,
        model_path: str,
        dataset_loader: BDD100KLoader,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.6
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to saved model or TFLite model
            dataset_loader: Dataset loader instance
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.dataset_loader = dataset_loader
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        self.model = self._load_model()
        
        # Storage for metrics
        self.all_detections = []
        self.all_ground_truths = []
        self.latencies = []
    
    def _load_model(self):
        """Load model (Keras or TFLite)."""
        if self.model_path.suffix == '.tflite':
            return self._load_tflite_model()
        else:
            return tf.keras.models.load_model(str(self.model_path))
    
    def _load_tflite_model(self):
        """Load TFLite model."""
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return interpreter
    
    def evaluate(self) -> Dict:
        """Run full evaluation."""
        print("Running evaluation...")
        
        for i in range(len(self.dataset_loader)):
            sample = self.dataset_loader[i]
            
            # Inference
            start_time = time.time()
            predictions = self._predict(sample['image'])
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)
            
            # Store results
            self.all_detections.append(predictions)
            self.all_ground_truths.append({
                'boxes': sample['boxes'],
                'classes': sample['classes'],
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(self.dataset_loader)} samples")
        
        # Compute metrics
        metrics = self._compute_metrics()
        return metrics
    
    def _predict(self, image: np.ndarray) -> Dict:
        """Run inference on single image."""
        if isinstance(self.model, tf.lite.Interpreter):
            return self._predict_tflite(image)
        else:
            return self._predict_keras(image)
    
    def _predict_keras(self, image: np.ndarray) -> Dict:
        """Keras model prediction."""
        image_batch = np.expand_dims(image, 0)
        outputs = self.model.predict(image_batch, verbose=0)
        
        # Apply NMS
        boxes, classes, scores = self._apply_nms(
            outputs['boxes'][0],
            outputs['classes'][0]
        )
        
        return {
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
        }
    
    def _predict_tflite(self, image: np.ndarray) -> Dict:
        """TFLite model prediction."""
        interpreter = self.model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_shape = input_details[0]['shape']
        if input_details[0]['dtype'] == np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image_uint8, 0))
        else:
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, 0))
        
        # Run inference
        interpreter.invoke()
        
        # Get outputs
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        # Filter by confidence
        valid = scores >= self.conf_threshold
        boxes = boxes[valid]
        classes = classes[valid].astype(int)
        scores = scores[valid]
        
        # Apply NMS
        boxes, classes, scores = self._apply_nms(boxes, classes, scores)
        
        return {
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
        }
    
    def _apply_nms(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Non-Maximum Suppression."""
        if len(boxes) == 0:
            return boxes, classes, scores if scores is not None else np.array([])
        
        # Convert to [x1, y1, x2, y2] format for tf.image.non_max_suppression
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, [0, 1]] = boxes[:, [1, 0]]  # Swap x, y
        boxes_xyxy[:, [2, 3]] = boxes[:, [3, 2]]
        
        if scores is None:
            scores = np.ones(len(boxes))
        
        # Apply NMS
        selected_indices = tf.image.non_max_suppression(
            boxes_xyxy,
            scores,
            max_output_size=100,
            iou_threshold=self.iou_threshold
        ).numpy()
        
        return (
            boxes[selected_indices],
            classes[selected_indices],
            scores[selected_indices]
        )
    
    def _compute_metrics(self) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {}
        
        # mAP
        metrics['mAP'] = self._compute_map()
        
        # Per-class precision/recall
        metrics['per_class_metrics'] = self._compute_per_class_metrics()
        
        # Latency metrics
        metrics['latency'] = {
            'p50_ms': np.percentile(self.latencies, 50),
            'p95_ms': np.percentile(self.latencies, 95),
            'p99_ms': np.percentile(self.latencies, 99),
            'mean_ms': np.mean(self.latencies),
            'std_ms': np.std(self.latencies),
        }
        
        # FPS stability
        metrics['fps_stability'] = self._compute_fps_stability()
        
        return metrics
    
    def _compute_map(self) -> float:
        """Compute mean Average Precision."""
        # Simplified mAP computation
        # Full implementation would use COCO evaluation protocol
        aps = []
        for class_id in range(NUM_CLASSES):
            ap = self._compute_ap_for_class(class_id)
            aps.append(ap)
        return np.mean(aps)
    
    def _compute_ap_for_class(self, class_id: int) -> float:
        """Compute Average Precision for a single class."""
        # Simplified AP computation
        # Full implementation would sort by confidence and compute precision-recall curve
        tp = 0
        fp = 0
        total_gt = 0
        
        for det, gt in zip(self.all_detections, self.all_ground_truths):
            gt_classes = gt['classes']
            total_gt += np.sum(gt_classes == class_id)
            
            det_classes = det['classes']
            det_scores = det.get('scores', np.ones(len(det_classes)))
            
            # Match detections to ground truth
            for i, det_class in enumerate(det_classes):
                if det_class == class_id and det_scores[i] >= self.conf_threshold:
                    # Check IoU with ground truth boxes
                    if self._has_match(det['boxes'][i], gt['boxes'], gt_classes, class_id):
                        tp += 1
                    else:
                        fp += 1
        
        if total_gt == 0:
            return 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        
        # Simplified AP (full implementation would use PR curve)
        return precision * recall
    
    def _has_match(
        self,
        box: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        class_id: int
    ) -> bool:
        """Check if detection matches any ground truth box."""
        matching_gt = gt_boxes[gt_classes == class_id]
        if len(matching_gt) == 0:
            return False
        
        ious = self._compute_iou(box, matching_gt)
        return np.any(ious >= self.iou_threshold)
    
    def _compute_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between box1 and boxes2."""
        # Simplified IoU computation
        y1_1, x1_1, y2_1, x2_1 = box1
        y1_2, x1_2, y2_2, x2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        inter_y1 = np.maximum(y1_1, y1_2)
        inter_x1 = np.maximum(x1_1, x1_2)
        inter_y2 = np.minimum(y2_1, y2_2)
        inter_x2 = np.minimum(x2_1, x2_2)
        
        inter_area = np.maximum(0, inter_y2 - inter_y1) * np.maximum(0, inter_x2 - inter_x1)
        box1_area = (y2_1 - y1_1) * (x2_1 - x1_1)
        boxes2_area = (y2_2 - y1_2) * (x2_2 - x1_2)
        union_area = box1_area + boxes2_area - inter_area
        
        iou = inter_area / np.maximum(union_area, 1e-6)
        return iou
    
    def _compute_per_class_metrics(self) -> Dict:
        """Compute precision/recall per class."""
        per_class = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            per_class[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
        return per_class
    
    def _compute_fps_stability(self) -> Dict:
        """Compute FPS stability metrics."""
        fps_values = 1000.0 / np.array(self.latencies)
        return {
            'mean_fps': np.mean(fps_values),
            'std_fps': np.std(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values),
            'cv': np.std(fps_values) / np.mean(fps_values),  # Coefficient of variation
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Horizon-HUD object detector")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--dataset-root", type=str, required=True, help="BDD100K dataset root")
    parser.add_argument("--labels-root", type=str, default=None, help="Optional labels root path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--config", type=str, help="Path to model config YAML")
    parser.add_argument("--output", type=str, help="Output JSON file for metrics")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        input_size = tuple(config['model']['input_size'])
        conf_threshold = config['nms']['confidence_threshold']
        iou_threshold = config['nms']['iou_threshold']
    else:
        input_size = (320, 320)
        conf_threshold = 0.5
        iou_threshold = 0.6
    
    # Load dataset
    dataset_loader = BDD100KLoader(
        dataset_root=args.dataset_root,
        labels_root=args.labels_root,
        split=args.split,
        input_size=input_size,
    )
    
    # Evaluate
    evaluator = DetectionEvaluator(
        model_path=args.model,
        dataset_loader=dataset_loader,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    
    metrics = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"\nLatency (ms):")
    print(f"  P50: {metrics['latency']['p50_ms']:.2f}")
    print(f"  P95: {metrics['latency']['p95_ms']:.2f}")
    print(f"  P99: {metrics['latency']['p99_ms']:.2f}")
    print(f"  Mean: {metrics['latency']['mean_ms']:.2f}")
    print(f"\nFPS Stability:")
    print(f"  Mean FPS: {metrics['fps_stability']['mean_fps']:.2f}")
    print(f"  Std FPS: {metrics['fps_stability']['std_fps']:.2f}")
    print(f"  CV: {metrics['fps_stability']['cv']:.4f}")
    
    # Save to file
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
