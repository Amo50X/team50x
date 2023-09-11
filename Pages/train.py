import streamlit as st
import cv2
from PIL import Image 
import pickle
import pathlib as pl
import os
import face_recognition

train_img_path = "training"
output_encode_path = pl.Path('output/encodings.pkl')

path = ""

createPath = False

def CreateData(path):
    image_placeholder = st.empty()
    st.write("Press Space to Take Picture")
    btn_Capture = st.button("Capture")
    cap = cv2.VideoCapture(0)
    while True:
        name = f"{path}/{len(os.listdir(path=path))}.jpg"
        suc, frame = cap.read()
        
        if btn_Capture:
            cv2.imwrite(name, frame)
            image_placeholder.empty()
            cap.release()
            break
        
        image_placeholder.image(frame, channels='BGR')

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()

def encode_faces(model: str = 'hog', encoding_location: pl.Path = output_encode_path) -> None:
    names = []
    encodings = []
    percent_complete = 0
    total_files = len(list(pl.Path(train_img_path).glob("*/*")))
    for index, filepath in enumerate(pl.Path(train_img_path).glob("*/*")):   
        percent_complete = round((index/(total_files-1)) * 100)
        name = filepath.parent.name
        img = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(img, model=model)
        face_encoding = face_recognition.face_encodings(img, face_locations)

        for encoding in face_encoding:
            names.append(name)
            encodings.append(encoding)

        loading_bar.progress(percent_complete, f"Encoding {name} images")
        percent_text.write(f'{percent_complete}%')


    name_encodings = {"names": names, "encodings": encodings}
    with encoding_location.open(mode='wb') as f:
        pickle.dump(name_encodings, f)  

st.title("Training")

st.subheader("Create Path for Data Images")

selectedPath = st.selectbox(label="Select a file", options=(os.listdir(train_img_path)))

col1,col2 = st.columns(2)
firstName = col1.text_input("First Name", max_chars=20)
lastName = col2.text_input("Last Name", max_chars=20)
if firstName and lastName:
    createPath = st.button("Create")

fullName = selectedPath

st.subheader(fullName if fullName else 'Create a Path')

if createPath:
    fullName = f"{firstName} {lastName}"
    path = os.path.join(train_img_path, fullName)
    print(path)
    if not os.path.exists(path):
        pl.Path(path).mkdir(exist_ok=True)
        st.success("Path was Successfully Created")
    else:
        st.success("Path Exist")

else:
    path = os.path.join(train_img_path, fullName)


takeOption = st.button("Take Images")


image_upload = st.file_uploader(label="Load Images", type=['.jpg', '.jpeg','.png'],accept_multiple_files=True)
if image_upload:
    for img in image_upload:
        with open(os.path.join(path, img.name),"wb") as f:
            f.write(img.getbuffer())
            
        

if takeOption:
    CreateData(path)

if os.listdir(path):                   
    images = os.listdir(path)
    index = st.number_input("Index", step=1, min_value=0,max_value=(len(images)-1))

    if images:
        image = Image.open(f"{path}/{images[index]}")
        st.image(image, use_column_width=True)

train_btn = st.button("Train")       

if train_btn:
    percent_text = st.empty()
    loading_bar = st.empty()
    encode_faces()

    
