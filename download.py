# # 1) SSD MobileNet V1 quantized COCO 300×300
# MODEL_ZIP_URL      = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
# MODEL_ZIP          = "coco_ssd_mobilenet_v1.zip"
# DETECT_MODEL_FILE  = "detect.tflite"
# LABELS_FILE        = "labelmap.txt"
#
# # 2) MoveNet SinglePose Lightning INT8
# POSE_MODEL_URL     = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
# POSE_MODEL_FILE    = "pose.tflite"



# def download_and_extract_detector():
#     if not (os.path.exists(DETECT_MODEL_FILE) and os.path.exists(LABELS_FILE)):
#         print("Downloading SSD-MobileNet V1 COCO model…")
#         urllib.request.urlretrieve(MODEL_ZIP_URL, MODEL_ZIP)
#         with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
#             for fname in zf.namelist():
#                 if fname.endswith(".tflite"):
#                     zf.extract(fname, ".")
#                     os.rename(fname, DETECT_MODEL_FILE)
#                 elif fname.endswith(".txt"):
#                     zf.extract(fname, ".")
#                     os.rename(fname, LABELS_FILE)
#         os.remove(MODEL_ZIP)
#
# def download_pose_model():
#     if not os.path.exists(POSE_MODEL_FILE):
#         print("Downloading MoveNet Lightning pose model (INT8)…")
#         resp = requests.get(POSE_MODEL_URL, allow_redirects=True)
#         if resp.status_code == 200 and resp.headers.get("Content-Type","").startswith("application/octet-stream"):
#             with open(POSE_MODEL_FILE, "wb") as f:
#                 f.write(resp.content)
#         else:
#             raise RuntimeError(f"Failed to download pose model (status {resp.status_code})")
#
# def prepare_models():
#     download_and_extract_detector()
#     download_pose_model()