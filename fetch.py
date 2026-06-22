from roboflow import Roboflow

API_KEY = "H6YmU5MTV67v2D5kmtq1"  # Use the key from your account
WORKSPACE = "kunal-vj4gg"
PROJECT = "mi6wrcis"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(1).download("yolov8")
print(f"✅ Dataset downloaded to: {dataset.location}")
