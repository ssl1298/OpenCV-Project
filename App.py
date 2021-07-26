################################################ Import Libraries ####################################################

import streamlit as st      ## --- Streamlit Library        (Streamlit Application)
import cv2                  ## --- openCV Library           (Computer Vision)
import numpy as np          ## --- Numpy Library            (Numerical Manipulations)
from PIL import Image       ## --- Pillow Library           (Image Rendering--default RGB) 
import mediapipe as mp      ## --- Mediapipe Library        (Face Detection,Meshing,Selfie Segmentation Solutions)
import face_recognition     ## --- Face Recognition Library (Face Recognition Manipulation)

######################################################################################################################

############################################# User Defined Functions #################################################

#------------------------------------------------- Home Page -----------------------------------------------------#

def HomePage():
    st.header("Image Manipulation using OpenCv - Python")
    st.subheader("Excited to experience the magic of OpenCV???")
    st.subheader("Let's Get Started!!!")
    #st.text("Scroll Down..")

    st.subheader("Instructions::")
    st.text("While Uploading Image, prefer the high quality, zoom and clear images for better results")
    st.text("Check out the below Images for reference..")

    col1, col2 = st.beta_columns(2)
    with col1:
        demo_img_file_path = 'Images/Robert.png'
        demo_image = np.array(Image.open(demo_img_file_path))
        demo_image = cv2.resize(demo_image,(300,400))
        st.image(demo_image,caption="Demo Image1")
    
    with col2:
        demo_img_file_path = 'Images/chris.png'
        demo_image = np.array(Image.open(demo_img_file_path))
        demo_image = cv2.resize(demo_image,(300,400))
        st.image(demo_image,caption="Demo Image2")

    st.subheader("Available Operations")
    
    expander_color_scheme = st.beta_expander(label='Color Schemes')
    with expander_color_scheme:
    
        st.header("Color Schemes Brief:")
        st.write("The Image will be rendered as colored scheme based on selection.")
        st.subheader("Example: Selection - GrayScale")
        #st.subheader("Selection - GrayScale")

        col3,col4 = st.beta_columns(2)
        with col3:
            st.subheader("Original Image")
            st.image(demo_image)

        with col4:
            st.subheader("GrayScale Image")
            demo_gray_img = Image.open('Images/Grayscale_Img.png')
            st.image(demo_gray_img)

    expander_face_detection = st.beta_expander(label='Face Detection')
    with expander_face_detection:
        st.subheader("Will be Updated Soon!!")


    expander_face_mesh = st.beta_expander(label='Face Mesh')
    with expander_face_mesh:
        st.subheader("Will be Updated Soon!!")


    expander_selfie_segmentation = st.beta_expander(label='Selfie Segmentation')
    with expander_selfie_segmentation:
        st.subheader("Will be Updated Soon!!")

    expander_face_recognition = st.beta_expander(label='Face Recognition')
    with expander_face_recognition:
        st.subheader("Will be Updated Soon!!")

#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

#---------------------------------------------- Color Schemes ----------------------------------------------------#

def ColorSchemes(option="Grayscale", image=None):
    if option == "Grayscale":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    else:
        zeros = np.zeros((image.shape[0],image.shape[1]),np.uint8)
        r,g,b = cv2.split(image)
        if option == "Blue":
            result = cv2.merge([zeros,zeros,b])
        elif option == "Green":
            result = cv2.merge([zeros,g,zeros])
        elif option == "Red":
            result = cv2.merge([r,zeros,zeros])
    
    return result

#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

#----------------------------------------------- Face Detection --------------------------------------------------#

def FaceDetection(image=None):

    #Face Detection Utility
    mp_face_detection = mp.solutions.face_detection
    #Face Drawing Utility
    mp_drawing = mp.solutions.drawing_utils

    model_face_detection = mp_face_detection.FaceDetection()

    results = model_face_detection.process(image)
    #print(results.detections)
    for landmarks in results.detections:
        mp_drawing.draw_detection(image,landmarks)
    
    return image
   
#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

#------------------------------------------------- Face Mesh -----------------------------------------------------#

def FaceMesh(image=None):
    
    #Face Meshing Utility
    mp_face_mesh = mp.solutions.face_mesh
    #Face Drawing Utility
    mp_drawing = mp.solutions.drawing_utils

    # Model facemash
    model_facemesh = mp_face_mesh.FaceMesh()

    drawing_spec = mp_drawing.DrawingSpec((0, 0, 255), thickness=1, circle_radius=1)

    results = model_facemesh.process(image)
    #print(results.multi_face_landmarks)
    
    for landmarks in results.multi_face_landmarks:
        
        #print(landmarks)
        mp_drawing.draw_landmarks(
            image=image,
			landmark_list=landmarks,
			connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    return image

#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

#------------------------------------------- Selfie Segmentation -------------------------------------------------#

def SelfieSegmentation(option,choice,image):
    
    #Face Drawing Utility
    mp_drawing = mp.solutions.drawing_utils
    #Background Change Utility
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    #Model for Segmentation
    model = mp_selfie_segmentation.SelfieSegmentation()

    #Background Image
    if option == 'Nature':
        bg_img_file_path  = 'Images/Nature.jpg'
        Bg_Image = np.array(Image.open(bg_img_file_path))

    elif option == 'Resort':
        bg_img_file_path  = 'Images/Resort.jpg'
        Bg_Image = np.array(Image.open(bg_img_file_path))

    elif option == 'Colors':
        
        Bg_Image = np.zeros(image.shape, dtype=np.uint8)
        if choice == 'Blue':
            Bg_Image[:] = (0,0,255)
        elif choice == 'Green':
            Bg_Image[:] = (0,255,0)
        elif choice == 'Red':
            Bg_Image[:] = (255,0,0)


    results = model.process(image)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    #print(results.segmentation_mask)
    #print(Bg_Image)

    if Bg_Image is None:
        Bg_Image = np.zeros(image.shape, dtype=np.uint8)
        Bg_Image[:] = (0,255,0)

    Bg_Image = cv2.resize(Bg_Image,(image.shape[1],image.shape[0]))
    blended_image = np.where(condition,image,Bg_Image)

    return blended_image

#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

#----------------------------------------------- Face Recognition ------------------------------------------------#

def FaceRecognition(image_train=None,image_test=None,text_input=''):

    #---2. Render the 128 Face Encoding Values 
    image_train_encoding = face_recognition.face_encodings(image_train)[0]
    #print("Image Training Encoding::",image_train_encoding)

    #---3. Render the 4 Face Locations [Top(0), Right(1), Bottom(2), Left(3)]
    image_train_face_locations = face_recognition.face_locations(image_train)[0]
    #print("TImage Train Face Locations::",image_train_face_locations)

    #---2. Render the 128 Face Encoding Values
    image_test_encoding = face_recognition.face_encodings(image_test)[0]
    #print("Image Test Encoding::",image_test_encoding)

    #---3. Render the 4 Face Locations [Top(0), Right(1), Bottom(2), Left(3)]
    image_test_face_locations = face_recognition.face_locations(image_test)[0]
    top, right, bottom, left = image_test_face_locations
    #print("TImage Train Face Locations::",image_train_face_locations)

    known_face_encodings = [
    image_train_encoding
    ]

    known_face_names = [
    text_input
    ]

    rgb_small_frame = cv2.resize(image_test, (0, 0), fx=1/4,fy=1/4)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # ---3. Comparision of Faces for Recognition [results --True/False]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        #---4. Render the distance between the train and test Image encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        #---5. If Match found ==> render the known face name
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        else:
            st.write(f"Could not recognize the face!!")
        face_names.append(name)
        #print(face_names)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
  
        #---6. Mask the Image Recognised
        cv2.rectangle(image_test, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image_test, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_test, name, (left + 6, bottom + 28), font, 1.0, (255, 255, 255), 1)    
        
    return image_test,matches

#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

#######################################################################################################################


################################################# Main Application ####################################################

# --- Application Title
st.title('Welcome - World of Computer Vision')

# --- Application Sidebar
add_selectbox = st.sidebar.selectbox(
    "Choose the operation to perform",
    ("Home","Color Schemes", "Face Detection", "Face Mesh", "Selfie Segmentation","Face Recognition")
)

# --- Manipulations based on the Selection
if add_selectbox == 'Home':
    
    # --- Function Call
    HomePage()


elif add_selectbox == 'Color Schemes':
    st.header("Color Schemes on Images")
    st.write("The Image uploaded will be rendered in applied color scheme.")
    st.subheader('To know more about the Color Schemes technique, Check out the Home Page!!')

    option = st.sidebar.radio(
        'Which Color Scheme you want your image to be in?',
        ('Grayscale', 'Red', 'Green','Blue'))
    
    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        # --- Function Call
        output_image = ColorSchemes(option,image)
        output_image = cv2.resize(output_image,(300,400))
        st.image(output_image,caption=f'{option} - Color Scheme Image')
        

elif add_selectbox == 'Face Detection':
    st.header("Face Detection from an Image")
    st.write("The face will be detected from the Image Uploaded.")
    st.subheader('To know more about the Face Detection Technique, Check out the Home Page!!')

    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        
        # --- Function Call
        output_image = FaceDetection(image)
        output_image = cv2.resize(output_image,(300,400))
        st.image(output_image)


elif add_selectbox == 'Face Mesh':
    st.header("Face Meshing from an image")
    st.write("The Face Mesh will be rendered for the face detected in an image.")
    st.subheader('To know more about the Face Mesh technique, Check out the Home Page!!')

    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        
        # --- Function Call
        output_image = FaceMesh(image)
        output_image = cv2.resize(output_image,(300,400))
        st.image(output_image)


elif add_selectbox == 'Selfie Segmentation':
    st.header("Selfie Segmentation for an Image")
    st.write("The required background will be blended for the image.")
    st.subheader('To know more about the Selfie Segmentation Technique, Check out the Home Page!!')

    option = st.sidebar.radio(
        'Which Background you want?',
        ('Nature','Resort', 'Colors'))

    choice = None
    if option == 'Colors':
        choice = st.sidebar.radio(
            'Which background color you want ?',
            ('Blue', 'Green','Red'))

    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)

        # --- Function Call
        output_image = SelfieSegmentation(option,choice,image)
        output_image = cv2.resize(output_image,(300,400))
        st.image(output_image)


elif add_selectbox == 'Face Recognition':
    st.header("Face Recognition from Images")
    st.write("The first image uploaded will be used for training the Face recognition Model.")
    st.write("Then second image would be used to test the model trained in recognising the face")
    st.subheader('To know more about the Face Recognition Technique, Check out the Home Page!!')

    text_input = st.sidebar.text_input(label='Enter name of the known face you want to upload:')
    st.write(f"Trained Model would recognise the Face : {text_input}")

    image_train_file_path = st.sidebar.file_uploader("Upload  1st image to train the model")
    
    if image_train_file_path is not None:
        image_train = np.array(Image.open(image_train_file_path))
        st.sidebar.image(image_train)

    image_test_file_path = st.sidebar.file_uploader("Upload  2nd image to test its Face Recognition capability")
    
    if image_test_file_path is not None:
        image_test = np.array(Image.open(image_test_file_path))
        st.sidebar.image(image_test)

    if image_train_file_path is not None and image_test_file_path is not None:
        
        # --- Function Call
        output_image , result= FaceRecognition(image_train,image_test,text_input)
        output_image = cv2.resize(output_image,(300,400))
        st.image(output_image)
            
 
#######################################################################################################################