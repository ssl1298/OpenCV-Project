# First OpenCV-Project
Computer Vision - Image Manipulations

Available Operations::
1. Color Schemes
2. Face Detection
3. Face Mesh
4. Face Recognition

Being a newbie in OpenCV python Library, building some application practically was a great experience.
***************************************************************************************************************************

Application Brief: @ http://computer-vision-world.herokuapp.com/
##################
Web Application : using Streamlit
Deployment Cloud Platform : Heroku
Programming Language : Python

*****************************************************************************************************************************
Repository Brief:
#################

You can run the application which is hosted using the HEROKU Platform @ http://computer-vision-world.herokuapp.com/
It consists of Home Page which displays the brief introduction about the available options.

You can always check out the Repository files for your reference:

>>> Images: Some set of images used in the application

>>> App.py: Actual python application file
------> This application is basically developed on streamlit- the better source for building up the Anaytics applications.
------> Then OpenCV library of python for performing CV related operations.

>>> Requirements.txt, Procfile, setup.sh files : for deploying the application on Heroku Platform

*******************************************************************************************************************************
Environmental Setup:
#####################

1. Programming Language: Python
2. Editor: VS-Code
3. Prefer to work on Virtual Environment for your specific application:: this helps in better management of Application setups
4. Install the libraries under the virtual environment using pip [pip install <libray_name>]
  i.    streamlit 
  ii.   opencv-python
  iii.  mediapipe
  iv.   cmake
  v.    dlib
  vi.   face_recognition
 
5. Having Git Bash and Heroku CLI would make things easy

********************************************************************************************************************************
References:
############

Python Installation : https://www.python.org/downloads/

VS-Code Installation : https://code.visualstudio.com/download

Virtual Environment Setup : https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/
>>> cd <path_for_creating_app_in_loca_sys[E:\Project\]>
>>> pip install virtualenv
>>> virtualenv <env_name>
>>> cd .\<env_name>\
>>> ls
>>> cd Scripts
>>> .\activate

Libraries Installation on Virtual Environment: https://www.youtube.com/watch?v=xaDJ5xnc8dc
>>> pip install opencv-python
>>> pip install streamlit
>>> pip install mediapipe
>>> pip install cmake
>>> pip install dlib ---- would take longer time to install[be patient]

----- Check all the libraries are installed successfully
>>> import cv2
>>> import streamlit as st
>>> import mediapipe as mp
>>> import face_recognition ---- to check the installtion of cmake and dlib

Deploying Streamlit Application on Heroku Platform: https://towardsdatascience.com/a-quick-tutorial-on-how-to-deploy-your-streamlit-app-to-heroku-874e1250dadd

*********************************************************************************************************************
Resources:

OpenCV Documentation : https://docs.opencv.org/4.5.2/

Mediapipe Documentation : https://google.github.io/mediapipe/

Streamlit Documentation : https://docs.streamlit.io/en/stable/

Application Essesntial : https://es.letsupgrade.in/facerec ////// https://youtu.be/XszmHbZislE
