"""
Loss functions for SSD object detection training.
"""

import tensorflow as tf
from tensorflow import keras


class SSDLoss(keras.losses.Loss):
    """
    SSD loss combining localization and classification losses.
    """
    
    def __init__(
        self,
        localization_weight: float = 1.0,
        classification_weight: float = 1.0,
        hard_example_mining: bool = True,
        neg_pos_ratio: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.localization_weight = localization_weight
        self.classification_weight = classification_weight
        self.hard_example_mining = hard_example_mining
        self.neg_pos_ratio = neg_pos_ratio
    
    def call(self, y_true, y_pred):
        """
        Compute SSD loss.
        
        Args:
            y_true: Dict with 'boxes' and 'classes'
            y_pred: Dict with 'boxes' and 'classes'
        
        Returns:
            Total loss
        """
        true_boxes = y_true['boxes']
        true_classes = y_true['classes']
        pred_boxes = y_pred['boxes']
        pred_classes = y_pred['classes']

        max_anchors = tf.shape(pred_boxes)[1]
        true_boxes = true_boxes[:, :max_anchors, :]
        true_classes = true_classes[:, :max_anchors]
        # Padded entries are marked with class -1.
        mask = tf.cast(true_classes >= 0, tf.float32)
        safe_true_classes = tf.where(true_classes < 0, tf.zeros_like(true_classes), true_classes)

        # Localization loss (Smooth L1) on valid anchors only.
        loc_loss = self._smooth_l1_loss(true_boxes, pred_boxes, mask)

        # Classification loss on valid anchors only.
        cls_loss = self._classification_loss(safe_true_classes, pred_classes, mask)
        
        total_loss = (
            self.localization_weight * loc_loss +
            self.classification_weight * cls_loss
        )
        
        return total_loss
    
    def _smooth_l1_loss(self, y_true, y_pred, mask):
        """Smooth L1 loss for bounding box regression."""
        diff = y_true - y_pred
        abs_diff = tf.abs(diff)
        smooth_l1 = tf.where(
            abs_diff < 1.0,
            0.5 * diff ** 2,
            abs_diff - 0.5
        )
        smooth_l1 = smooth_l1 * tf.expand_dims(mask, axis=-1)
        denom = tf.maximum(tf.reduce_sum(mask) * 4.0, 1.0)
        return tf.reduce_sum(smooth_l1) / denom
    
    def _classification_loss(
        self,
        y_true,
        y_pred,
        mask
    ):
        """Masked classification loss on valid anchors."""
        # Cross-entropy loss
        ce_loss = keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=False
        )
        ce_loss = ce_loss * mask
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(ce_loss) / denom
