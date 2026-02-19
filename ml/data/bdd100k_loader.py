"""
BDD100K Dataset Loader for Horizon-HUD General Object Detection.
Handles dataset loading, preprocessing, and class mapping.
"""

import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2

from ml.utils.class_mapping import map_bdd100k_to_horizon, get_class_id


class BDD100KLoader:
    """Loader for BDD100K detection dataset."""
    
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        target_classes: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (320, 320),
        preserve_diversity: bool = True,
        labels_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize BDD100K loader.
        
        Args:
            dataset_root: Path to BDD100K dataset root
            split: Dataset split ('train', 'val', 'test')
            target_classes: List of target class names (None = all)
            input_size: Target input size (height, width)
            preserve_diversity: Preserve day/night/weather diversity
            labels_root: Optional separate path for annotation JSON files
            max_samples: Optional cap on loaded samples for quick sanity runs
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.input_size = input_size
        self.preserve_diversity = preserve_diversity
        self.labels_root = Path(labels_root) if labels_root else self.dataset_root / "labels"
        self.max_samples = max_samples

        self.images_dir = self._resolve_images_dir()
        self.labels_file = self._resolve_labels_file()
        self.labels_dir = self.labels_root / self.split
        self.annotation_mode = self._resolve_annotation_mode()
        
        # Load annotations
        self.annotations = self._load_annotations()
        self.samples = self._prepare_samples(target_classes)

    def _resolve_images_dir(self) -> Path:
        """Resolve image directory for either BDD100K or flat split layout."""
        bdd_images_dir = self.dataset_root / "images" / "100k" / self.split
        split_dir = self.dataset_root / self.split
        if bdd_images_dir.exists():
            return bdd_images_dir
        return split_dir

    def _resolve_labels_file(self) -> Path:
        """Resolve labels JSON path, supporting common BDD naming patterns."""
        candidates = [
            self.labels_root / f"det_{self.split}.json",
            self.labels_root / f"{self.split}.json",
            self.labels_root / f"instances_{self.split}.json",
            self.labels_root / "annotations" / f"det_{self.split}.json",
            self.labels_root / "annotations" / f"instances_{self.split}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_annotation_mode(self) -> str:
        """Pick annotation format based on available files."""
        if self.labels_file.exists():
            return "bdd_aggregate"
        if self.labels_dir.exists():
            return "per_image_json"
        return "missing"
        
    def _load_annotations(self) -> List[Dict]:
        """Load BDD100K JSON annotations."""
        if self.annotation_mode == "bdd_aggregate":
            with open(self.labels_file, 'r') as f:
                return json.load(f)

        if self.annotation_mode == "per_image_json":
            return sorted(self.labels_dir.glob("*.json"))

        if self.annotation_mode == "missing":
            raise FileNotFoundError(
                "Labels file not found.\n"
                f"Checked expected path: {self.labels_file}\n"
                f"Dataset root: {self.dataset_root}\n"
                f"Labels root: {self.labels_root}\n"
                "Pass --labels-root if labels are stored in another folder."
            )

        raise RuntimeError(f"Unsupported annotation mode: {self.annotation_mode}")
    
    def _prepare_samples(
        self,
        target_classes: Optional[List[str]]
    ) -> List[Dict]:
        """Prepare samples with class mapping and filtering."""
        if self.annotation_mode == "bdd_aggregate":
            return self._prepare_samples_bdd_aggregate(target_classes)
        if self.annotation_mode == "per_image_json":
            return self._prepare_samples_per_image_json(target_classes)
        return []

    def _prepare_samples_bdd_aggregate(
        self,
        target_classes: Optional[List[str]]
    ) -> List[Dict]:
        """Prepare samples from BDD aggregate split JSON."""
        samples = []
        
        for ann in self.annotations:
            image_name = ann['name']
            image_path = self.images_dir / image_name
            
            if not image_path.exists():
                continue
            
            # Extract attributes for diversity preservation
            attributes = ann.get('attributes', {})
            weather = attributes.get('weather', 'unknown')
            timeofday = attributes.get('timeofday', 'unknown')
            
            # Map and filter labels
            boxes = []
            classes = []
            
            for label in ann.get('labels', []):
                bdd_class = label.get('category', '').lower()
                horizon_class = map_bdd100k_to_horizon(bdd_class)
                
                if horizon_class is None:
                    continue
                
                if target_classes and horizon_class not in target_classes:
                    continue
                
                # Extract bounding box
                box2d = label.get('box2d', {})
                if not box2d:
                    continue
                
                x1 = box2d.get('x1', 0)
                y1 = box2d.get('y1', 0)
                x2 = box2d.get('x2', 0)
                y2 = box2d.get('y2', 0)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                boxes.append([y1, x1, y2, x2])  # TF format: [ymin, xmin, ymax, xmax]
                classes.append(get_class_id(horizon_class))
            
            if len(boxes) > 0:
                samples.append({
                    'image_path': str(image_path),
                    'boxes': np.array(boxes, dtype=np.float32),
                    'classes': np.array(classes, dtype=np.int32),
                    'weather': weather,
                    'timeofday': timeofday,
                })
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    break
        
        return samples

    def _resolve_image_path_for_stem(self, stem: str) -> Optional[Path]:
        """Resolve image path from label stem across common extensions."""
        candidates = [
            self.images_dir / f"{stem}.jpg",
            self.images_dir / f"{stem}.jpeg",
            self.images_dir / f"{stem}.png",
            self.images_dir / stem,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _prepare_samples_per_image_json(
        self,
        target_classes: Optional[List[str]]
    ) -> List[Dict]:
        """Prepare samples from per-image BDD JSON files (one JSON per image)."""
        samples = []

        for label_path in self.annotations:
            with open(label_path, "r") as f:
                ann = json.load(f)

            image_stem = ann.get("name", label_path.stem)
            image_path = self._resolve_image_path_for_stem(image_stem)
            if image_path is None:
                continue

            frames = ann.get("frames", [])
            if not frames:
                continue
            objects = frames[0].get("objects", [])

            boxes = []
            classes = []
            for obj in objects:
                bdd_class = obj.get("category", "").lower()
                horizon_class = map_bdd100k_to_horizon(bdd_class)
                if horizon_class is None:
                    continue
                if target_classes and horizon_class not in target_classes:
                    continue

                box2d = obj.get("box2d", {})
                if not box2d:
                    continue

                x1 = box2d.get("x1", 0)
                y1 = box2d.get("y1", 0)
                x2 = box2d.get("x2", 0)
                y2 = box2d.get("y2", 0)
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([y1, x1, y2, x2])
                classes.append(get_class_id(horizon_class))

            if boxes:
                samples.append({
                    "image_path": str(image_path),
                    "boxes": np.array(boxes, dtype=np.float32),
                    "classes": np.array(classes, dtype=np.int32),
                    "weather": "unknown",
                    "timeofday": "unknown",
                })
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    break

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Resize image
        image_resized = cv2.resize(image, self.input_size[::-1])
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Scale bounding boxes
        scale_y = self.input_size[0] / original_shape[0]
        scale_x = self.input_size[1] / original_shape[1]
        
        boxes_scaled = sample['boxes'].copy()
        boxes_scaled[:, [0, 2]] *= scale_y  # y coordinates
        boxes_scaled[:, [1, 3]] *= scale_x  # x coordinates
        
        # Normalize boxes to [0, 1]
        boxes_normalized = boxes_scaled.copy()
        boxes_normalized[:, [0, 2]] /= self.input_size[0]
        boxes_normalized[:, [1, 3]] /= self.input_size[1]
        
        return {
            'image': image_normalized,
            'boxes': boxes_normalized,
            'classes': sample['classes'],
            'num_detections': len(sample['classes']),
        }
    
    def create_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
        max_boxes: int = 1000,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset."""
        max_boxes = max(1, int(max_boxes))

        def generator():
            indices = list(range(len(self.samples)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                sample = self[idx]
                # Keep a fixed upper bound for batching/training.
                if sample['num_detections'] > max_boxes:
                    sample['boxes'] = sample['boxes'][:max_boxes]
                    sample['classes'] = sample['classes'][:max_boxes]
                    sample['num_detections'] = max_boxes
                yield sample
        
        output_signature = {
            'image': tf.TensorSpec(shape=(*self.input_size, 3), dtype=tf.float32),
            'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'classes': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'num_detections': tf.TensorSpec(shape=(), dtype=tf.int32),
        }
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if augment:
            dataset = dataset.map(self._augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'image': (*self.input_size, 3),
                'boxes': (max_boxes, 4),
                'classes': (max_boxes,),
                'num_detections': (),
            },
            padding_values={
                'image': tf.constant(0.0, dtype=tf.float32),
                'boxes': tf.constant(0.0, dtype=tf.float32),
                'classes': tf.constant(-1, dtype=tf.int32),
                'num_detections': tf.constant(0, dtype=tf.int32),
            },
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def _augment_fn(self, sample):
        """Apply augmentations."""
        image = sample['image']
        boxes = sample['boxes']
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.3)
        
        # Random scale (with box adjustment)
        scale = tf.random.uniform([], 0.8, 1.2)
        h, w = self.input_size
        new_h = tf.cast(h * scale, tf.int32)
        new_w = tf.cast(w * scale, tf.int32)
        
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(image, h, w)
        
        # Box scaling would need adjustment here (simplified)
        
        return {
            'image': image,
            'boxes': boxes,
            'classes': sample['classes'],
            'num_detections': sample['num_detections'],
        }
