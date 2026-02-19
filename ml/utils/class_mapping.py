"""
Class mapping utilities for Horizon-HUD General Object Detection.
Maps BDD100K classes to target classes with stable IDs.
"""

# Stable class IDs for Horizon-HUD (DO NOT CHANGE - breaks downstream pipeline)
CLASS_ID_MAPPING = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "road_obstacle": 3,
}

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist", "road_obstacle"]
NUM_CLASSES = len(CLASS_NAMES)

# BDD100K to Horizon-HUD mapping
BDD100K_TO_HORIZON = {
    # Vehicle classes
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "rider": "vehicle",  # Motorcycle riders count as vehicles
    
    # Pedestrian
    "person": "pedestrian",
    
    # Cyclist
    "bike": "cyclist",
    "motor": "cyclist",  # Motorcycles
    
    # Road obstacles
    "traffic sign": "road_obstacle",
    "traffic light": "road_obstacle",
    "pole": "road_obstacle",
    "traffic cone": "road_obstacle",
}

def map_bdd100k_to_horizon(bdd100k_class: str) -> str:
    """Map BDD100K class name to Horizon-HUD class."""
    return BDD100K_TO_HORIZON.get(bdd100k_class.lower(), None)

def get_class_id(class_name: str) -> int:
    """Get stable class ID for Horizon-HUD class."""
    return CLASS_ID_MAPPING.get(class_name.lower(), -1)

def get_class_name(class_id: int) -> str:
    """Get class name from stable class ID."""
    if 0 <= class_id < NUM_CLASSES:
        return CLASS_NAMES[class_id]
    return "unknown"

def create_labelmap_file(output_path: str):
    """Create labelmap.txt file compatible with TFLite detection models."""
    with open(output_path, 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
