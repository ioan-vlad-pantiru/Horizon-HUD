import cv2
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter

# --- Load Labels --- M
def load_labels(filename):
    with open(filename, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}
# -------------------

# Load model
interpreter = Interpreter(model_path="models/1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(f"Output details: {output_details}") # Removed print statement

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Load the full ImageNet labels
labels = load_labels('imagenet_labels.txt')
if not labels:
    print("Error: Could not load labels from imagenet_labels.txt")
    exit()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input
    input_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_frame.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Process classification output
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Get the single output tensor
    top_class_index = np.argmax(output_data)  # Find the index with the highest probability
    confidence = output_data[top_class_index] # Get the confidence score for the top class

    # Map index to label (adjust label map if needed)
    # Assuming ImageNet-based labels where index might need adjustment if background is 0
    # Check your specific model's label mapping. For now, assume direct mapping or adjust index.
    # If your model uses indices 1-1000 for classes, use top_class_index.
    # If it uses 0-999, use top_class_index. If it includes background at 0, adjust accordingly.
    # Let's try direct mapping first. If labels seem off, we might need to adjust.
    predicted_label = labels.get(top_class_index, f'Unknown index: {top_class_index}')

    # Display the top prediction
    display_text = f"{predicted_label} ({confidence:.2f})"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show HUD window
    cv2.imshow("HUD Classification", frame) # Renamed window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()