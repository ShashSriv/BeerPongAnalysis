from ultralytics import YOLO


# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
image = "images/Cup.jpg"

# Run inference on the source
results = model(image)  # list of Results objects