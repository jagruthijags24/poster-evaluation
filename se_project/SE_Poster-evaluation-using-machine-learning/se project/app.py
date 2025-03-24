import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from PIL import Image
import os
import numpy as np
from flask import Flask, request, render_template
import cv2
from cv2 import cvtColor, Laplacian, COLOR_BGR2GRAY,Canny
import pytesseract 
from io import BytesIO
from fontTools.ttLib import TTFont
import re
import requests
from pyzbar.pyzbar import decode

app = Flask(__name__)

# Define the folder for uploading posters
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the "uploads" directory exists, or create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to calculate average RGB values
def calculate_average_rgb(image_path):
    with Image.open(image_path) as img:
        converted_img = img.convert("RGB")
        r, g, b = converted_img.split()
        
        r_sum = sum(r.getdata())
        g_sum = sum(g.getdata())
        b_sum = sum(b.getdata())
        
        r_avg = r_sum // len(r.getdata())
        g_avg = g_sum // len(g.getdata())
        b_avg = b_sum // len(b.getdata())
    
    return r_avg, g_avg, b_avg

def evaluate_indentation(image_path):
    # In this example, we'll use a simple threshold-based evaluation
    img = Image.open(image_path)
    img_data = np.array(img)
    threshold = 150  # Adjust this threshold as needed
    indentation_score = (np.mean(img_data) > threshold)  # Simulated result
    return indentation_score

def analyze_poster_size_and_dimension(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def analyze_image_clarity(image_path):
    img = Image.open(image_path)
    if img.mode == 'RGB':
        img = cvtColor(np.array(img), COLOR_BGR2GRAY)
    clarity_score = Laplacian(img, cv2.CV_64F).var()
    return clarity_score

def analyze_clutter(image_path):
    img = Image.open(image_path)
    img = cvtColor(np.array(img), COLOR_BGR2GRAY)

    # Use edge detection to identify edges in the image
    edges = Canny(img, threshold1=100, threshold2=200)

    # Count the number of edge pixels as a measure of clutter
    clutter_score = np.count_nonzero(edges)
    
    return clutter_score

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Function to validate a URL
def validate_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def qr_code_detector(image_path):
    image = Image.open(image_path)
    decoded_objects = decode(image)
    if decoded_objects:
        for obj in decoded_objects:
            text = obj.data.decode('utf-8')
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

            if re.search(url_pattern, text):
                return True, obj.data.decode('utf-8')
            return False, obj.data.decode('utf-8')
    else:
        return False, "No qr code found"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp'}


@app.route("/", methods=["GET", "POST"])
def index():
    average_rgb_result = ""  # Initialize with an empty string
    indentation_result = ""
    size_dimension_result = ""
    clarity_result = ""
    clutter_result = ""
    qr_code_image = None
    extracted_text = ""
    barcode_results = []
    detected_urls = []
    url_validations = []
    show_results = False
    
    if request.method == "POST":
        # Check if the post request has a file part
        if 'poster' not in request.files:
            return "No file part"

        file = request.files['poster']

        if file.filename == '' or not allowed_file(file.filename):
            return "Invalid or no selected file"

        if file and allowed_file(file.filename):
            # Save the uploaded poster
            poster_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(poster_path)

            # Calculate average RGB values
            r, g, b = calculate_average_rgb(poster_path)
            average_rgb_result = f" ({r}, {g}, {b})"

            # Evaluate indentation (replace with actual model prediction)
            indentation_score = evaluate_indentation(poster_path)
            indentation_result = f"{indentation_score}"

            clarity_score = analyze_image_clarity(poster_path)
            clarity_result = f" {clarity_score}"

            width, height = analyze_poster_size_and_dimension(poster_path)
            size_dimension_result = f" {width}x{height}"
            
            clutter_score = analyze_clutter(poster_path)
            clutter_result = f" {clutter_score}"
            
            extracted = extract_text_from_image(poster_path)
            extracted_text = f" {extracted}"
            
            
            has_qr_code, qr_code_data = qr_code_detector(poster_path)
            qr_data = qr_code_data
            if has_qr_code:
                qr_data = f"<a href='{qr_code_data}'>{qr_code_data}</a>"
            
            
            return render_template('results.html', 
                                   average_rgb_result=average_rgb_result,
                                   indentation_result=indentation_result,
                                   size_dimension_result=size_dimension_result,
                                   clarity_result=clarity_result,
                                   clutter_result=clutter_result,
                                   extracted_text=extracted_text,
                                   qr_code_data=qr_code_data)
            
            

            #return f"{average_rgb_result}<br>{indentation_result}<br>{size_dimension_result}<br>{clarity_result}<br>{clutter_result}<br>{extracted_text}<br>QR code data: {qr_data}"
        

    return render_template('index.html', 
               average_rgb_result=average_rgb_result,
               indentation_result=indentation_result,
               size_dimension_result=size_dimension_result,
               clarity_result=clarity_result,
               clutter_result=clutter_result,
               extracted_text=extracted_text,
               show_results=True)

if __name__ == '__main__':
    app.run(debug=True)

