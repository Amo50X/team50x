import streamlit as st
import cv2
import face_recognition
import pathlib as pl
import pickle
from collections import Counter
import os
import numpy as np
# from ultralytics import YOLO
# import supervision as sv

# MODEL = 'weights/yolov8s.pt'

# model = YOLO(MODEL)

output_encode_path = pl.Path('output/encodings.pkl')
train_img_path = "training"

class_list = os.listdir(train_img_path)
attend_check = [False for i in class_list]


st.set_page_config(
    page_title="Face Recognition",
    page_icon="🧑‍🦱",
)

st.title("Main Page")
st.sidebar.success("Select a page above")

cap = cv2.VideoCapture(0)

image_placeholder = st.empty()
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

with output_encode_path.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
empty_list = [st.empty() for i in class_list]
count = 0
while cap.isOpened():
    
    ret, frame = cap.read()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_location = face_recognition.face_locations(rgb_frame)
    
    if len(face_location) > 0:
        face_location1 = face_location[0]
        y1,x1,y2,x2 = face_location1[0], face_location1[1], face_location1[2], face_location1[3]

        face_encoded = face_recognition.face_encodings(rgb_frame, face_location)

        for bounding_box, unknown_encoding in zip(face_location, face_encoded):
            name = _recognize_face(unknown_encoding, loaded_encodings)
            # y1,x1,y2,x2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
            cv2.putText(rgb_frame, name, (x1-(100),y1), cv2.FONT_HERSHEY_DUPLEX, 1 , (0,0,0), 2)
            cv2.rectangle(rgb_frame, (x1,y1), (x2,y2),(0,255,0), 2)

            for index ,learn in enumerate(class_list):
                if name == learn:
                    attend_check[index] = True
                    pass

    for index ,mark in enumerate(class_list):
        empty_list[index].toggle(label=mark, value=attend_check[index], key=count)
        count +=1         

    image_placeholder.image(caption="Face detection", image=rgb_frame, channels="RGB")
    
# def attendence():
