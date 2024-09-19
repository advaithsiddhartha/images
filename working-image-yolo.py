import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

model = YOLO('yolov8n.pt')

def process_image(input_image):
    img = np.array(input_image)

    if img.ndim == 3 and img.shape[2] == 4:  
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.ndim == 2:  
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    results = model(img)

    annotated_img = results[0].plot()  

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    vehicle_count = len(results[0].boxes) 

    class_ids = results[0].boxes.cls.cpu().numpy() 
    class_names = [results[0].names[int(cls_id)] for cls_id in class_ids]  

    return annotated_img_rgb, vehicle_count, class_names

st.title("Vehicle Detection and Counting App for Image inputs")
st.info("`TEAM SIRIUS _ SMART INDIA HACKATHON`")
st.success("R. Krishna Advaith Siddhartha , S. Ravi Teja R. Bhoomika    , V. Subhash    , K. Nakshatra    , M. Abhinav")
st.write("Upload an image to detect vehicles and categorize them.")
st.markdown("""*We have tried using images as our inputs in the first stage , and these are the results we have obtained . We have also presented our testing programme which is requested to be kept confidential. We have tried integrating our idea into a streamlit app , that is easy and efficient to deploy ( as a simulation ). 
This Streamlit app performs vehicle detection and counting using the YOLOv8 model, which is one of the state-of-the-art object detection architectures. The app is designed to allow users to upload images and get results with vehicle counts and types in real time.*
                   
**Library Setup:**
            
The app starts by importing necessary libraries such as streamlit for building the web interface, ultralytics for accessing the YOLOv8 model, and PIL (Python Imaging Library) to handle image upload and processing. OpenCV is used for image manipulation.

**Loading YOLO Model:**
            
We load the YOLOv8 model using YOLO('yolov8n.pt'). The YOLO model is a pre-trained object detection model capable of detecting a wide variety of objects, including vehicles. The model will return bounding boxes for each detected object in the image, along with their class labels (such as car, truck, bus, etc.).*""")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    input_image = Image.open(uploaded_image)
    st.image(input_image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Vehicles"):
        with st.spinner("Processing..."):
            processed_image, vehicle_count, detected_categories = process_image(input_image)

            st.image(processed_image, caption='Processed Image', use_column_width=True)
            st.write(f"Total number of vehicles detected: {vehicle_count}")
            st.write("Detected vehicle categories:", detected_categories)
