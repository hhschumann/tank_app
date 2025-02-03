from tankbuster import bust
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def draw_text_with_background(frame, text, position, font, font_scale, text_color, bg_color, thickness=2):
    # Text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness) 

    # Calculate background coordinates
    x, y = position
    background_top_left = (x, y - text_height - 5)
    background_bottom_right = (x + text_width, y + 5)

    # Draw background rectangle
    cv2.rectangle(frame, background_top_left, background_bottom_right, bg_color, cv2.FILLED)

    # Draw text over the background
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def process_image(image_path):
    frame = cv2.imread(image_path)
    unannotated_frame = frame.copy()
    result = bust(image_path, network="ResNet")

    max_key = max(result, key=result.get)
    max_value = result[max_key].astype(float) * 100

    height, width, _ = frame.shape

    font_scale = min(width, height) / 500  
    thickness = max(1, int(font_scale * 2))
    position = (int(width * 0.02), int(height * 0.05)) 

    draw_text_with_background(frame, f"{max_key} - {round(max_value,2)}%", position,
                                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255,), (118, 29, 14, 0.79), thickness)
    org_frame.image(unannotated_frame, channels="BGR")
    ann_frame.image(frame, channels="BGR")

if __name__=="__main__":
    st.title("Tank Classification Demo")

    input_file = st.sidebar.file_uploader("Upload Image File", type=["jpeg", "jpg", "png"])
    if input_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
            temp_file.write(input_file.read())
            input_file_path = temp_file.name

    if st.button('Run Model'):
        col1, col2 = st.columns(2)
        org_frame = col1.empty()
        ann_frame = col2.empty()
        process_image(input_file_path)
    
    
        
