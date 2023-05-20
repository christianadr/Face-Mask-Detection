import streamlit as st
from PIL import Image, ImageDraw
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')

def detection(image):
    """ Run detection on uploaded image """
    results = model([image])
    return results

def draw_bounding_boxes(image, boxes, labels, confidences):
    """ Drawing bounding boxes on detected objects on image 
        whether it has facemask or none """
    draw = ImageDraw.Draw(image)
    for box, label, confidence in zip(boxes, labels, confidences):
        color = 'red' if label == 'NO-Mask' else 'green'  # Use red for no_mask, green for mask
        draw.rectangle(box, outline=color, width=5)
        draw.text((box[0], box[1] - 20), f"{label}: {confidence:.2f}", fill=color)
    return image

def main():
    """ Displaying window using Streamlit API """
    st.header("Face Mask Detection using YOLOv5")
    st.write("""
    ##### Christian Ainsley Del Rosario | CPE32S6
    **Green boxes** - With Face Mask | **Red boxes** - Without Face Mask
    """)

    img = st.file_uploader("Choose an image from your computer", type=['jpg', 'png'])

    col1, col2 = st.columns(2)
    if img is not None:
        with col1:
            image = Image.open(img)
            st.image(image, caption='Uploaded image', use_column_width=True)
        with col2:
            image = Image.open(img)
            results = detection(image)
            boxes = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = results.pandas().xyxy[0]['name'].tolist()
            confidences = results.pandas().xyxy[0]['confidence'].tolist()

            # Display the image with bounding boxes and labels
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes, labels, confidences)
            st.image(image_with_boxes, caption='With bounding boxes', use_column_width=True)

        st.success("Image successfully detected!")
    else:
        default_image = 'assets/download.jpg'
        with col1:
            st.image(default_image, caption='Sample Image', use_column_width=True)
        with col2:
            image = Image.open(default_image)
            results = detection(image)
            boxes = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = results.pandas().xyxy[0]['name'].tolist()
            confidences = results.pandas().xyxy[0]['confidence'].tolist()

            # Display the image with bounding boxes and labels
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes, labels, confidences)
            st.image(image_with_boxes, caption='With bounding boxes', use_column_width=True)

        st.write("You can upload an image consisting of people with or without face mask")

    st.write(
        """
        The following link will redirect you to the Colab where the training of the model occurs: [Google Colab Link](https://colab.research.google.com/drive/1HzFrpWmx2o9AkPd9she6oArFm6cwN_ra?usp=sharing)
        """
    )

if __name__ == '__main__':
    main()
