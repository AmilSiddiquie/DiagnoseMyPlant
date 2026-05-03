import os
import time # Added this for the cleanup timer!
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
# Path to your model inside the 'model' folder
MODEL_PATH = os.path.join('model', 'leaf_model.h5') 
UPLOAD_FOLDER = os.path.join('static', 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the leaf disease CNN model
model = load_model(MODEL_PATH)

# --- GEMINI AI CONFIGURATION ---
# SECURED FOR DEPLOYMENT: The key is now hidden and fetched from Hugging Face Secrets.
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel('gemini-2.5-flash')

# Your exact 39 classes from the testing script
CLASSES = [
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy', 
    'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Orange___healthy', 
    'Peach___Bacterial_spot', 
    'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# --- IMAGE CLEANUP FUNCTION ---
def cleanup_old_images():
    """Deletes uploaded leaf images older than 2 hours to save server space."""
    SAFE_FILES = ['amil.jpeg', 'uzair.jpeg', 'ehtesham.jpeg', 'shahid.jpeg']
    now = time.time()
    
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename in SAFE_FILES:
                continue # Skip team photos!
                
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                # Check if the file is older than 2 hours (7200 seconds)
                if os.stat(file_path).st_mtime < now - 7200:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Could not delete {filename}: {e}")

def prepare_image(img_path):
    # 1. Read the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to 128x128 (as per your training)
    img = cv2.resize(img, (128, 128))
    
    # 3. Convert to float32 BUT DO NOT DIVIDE BY 255 
    # unless you are 100% sure you did that in training.
    input_arr = np.array([img]).astype('float32') 
    
    return input_arr

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Clean up old images before processing a new one!
    cleanup_old_images()

    if 'file' not in request.files: return "No file"
    file = request.files['file']
    if file.filename == '': return "No file"

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Preprocess using the 128x128 logic
            input_arr = prepare_image(filepath)
            
            # Predict
            prediction = model.predict(input_arr)
            result_index = np.argmax(prediction)
            model_prediction = CLASSES[result_index]

            # Get Management Advice
            advice = get_management_advice(model_prediction)

            return render_template('index.html', 
                                 prediction=model_prediction, 
                                 advice=advice,
                                 image_path=f"static/uploads/{filename}")
        except Exception as e:
            return f"Error: {str(e)}"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        
        # Give the AI its instructions so it acts like an agricultural expert
        prompt = f"""
        You are an expert agricultural AI assistant named LeafAI Agronomist. 
        Keep your answers concise, helpful, and focused on plant care, crop diseases, fertilizers, and farming techniques.
        Do not use markdown formatting like asterisks or bold text, just provide clean, readable text.
        
        User asks: {user_message}
        """
        
        # Generate the response using your API key
        response = chat_model.generate_content(prompt)
        return jsonify({"reply": response.text})
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"reply": "I'm having a little trouble connecting to the knowledge base right now. Please try asking again!"})

def get_management_advice(label):
    label_lower = label.lower()

    # 1. Healthy Plant
    if "healthy" in label_lower:
        return "🎉 CONGRATULATIONS! Your plant is perfectly healthy. Keep up the excellent work with your current watering, soil management, and sunlight routine!"

    # 2. Viral Infections (Tomato Mosaic, Yellow Leaf Curl)
    elif "virus" in label_lower:
        return "🦠 WHY IT HAPPENS: Viral infections are usually transmitted by sap-sucking pests (like aphids or whiteflies) or through contaminated pruning tools. WHEN IT HAPPENS: Most common during peak insect season in mid-to-late summer. SOLUTION: Unfortunately, there is no chemical cure for plant viruses. Immediately remove and destroy the infected plants to save your healthy crops. Use insecticidal soap to control pest populations."

    # 3. Late Blight
    elif "late_blight" in label_lower or "late blight" in label_lower:
        return "🍂 WHY IT HAPPENS: Caused by the water mold Phytophthora infestans, which spreads rapidly through airborne spores. WHEN IT HAPPENS: Thrives in cool, consistently wet, and highly humid weather. SOLUTION: Remove and destroy infected leaves immediately. Apply a copper-based fungicide to protect healthy foliage, and ensure good spacing between plants to improve airflow."

    # 4. Early Blight & Leaf Spots (Cercospora, Septoria, etc.)
    elif "early_blight" in label_lower or "spot" in label_lower or "early blight" in label_lower:
        return "🍂 WHY IT HAPPENS: Caused by soil-borne fungi that splash onto the lower leaves during heavy watering or rain. WHEN IT HAPPENS: Mostly during warm, humid weather or periods of heavy rainfall. SOLUTION: Prune off affected lower leaves. Water the soil directly at the base (avoid wetting leaves) and apply a thick mulch layer to prevent soil splashing. Use a broad-spectrum fungicide if the spread continues."

    # 5. Rust Diseases (Cedar Apple Rust, Common Rust)
    elif "rust" in label_lower:
        return "🟠 WHY IT HAPPENS: Fungal spores that require a living host to survive, easily spreading via wind and water splash. WHEN IT HAPPENS: Usually develops rapidly in mild, moist spring weather. SOLUTION: Remove severely infected leaves. Avoid overhead watering. Apply sulfur or copper-based fungicides early in the season as a preventative measure."

    # 6. Powdery Mildew
    elif "mildew" in label_lower:
        return "💨 WHY IT HAPPENS: A surface fungal disease that targets dry foliage in high-humidity environments. WHEN IT HAPPENS: Very common in late summer and early fall when days are warm but nights are cool and damp. SOLUTION: Improve air circulation by thinning the plant. Apply neem oil or a baking soda spray (1 tbsp baking soda + 1 gallon water + a few drops of liquid soap)."

    # 7. Spider Mites
    elif "mite" in label_lower:
        return "🕷️ WHY IT HAPPENS: Spider mites are microscopic arachnids that suck sap directly from plant cells, causing yellow stippling. WHEN IT HAPPENS: Their populations explode during very hot, dry, and dusty weather. SOLUTION: Spray the plant forcefully with a hose to physically knock them off. Apply neem oil or insecticidal soap, and keep the area slightly more humid."

    # 8. Citrus Greening
    elif "greening" in label_lower:
        return "🍋 WHY IT HAPPENS: A severe bacterial disease spread by a tiny insect called the Asian citrus psyllid. WHEN IT HAPPENS: Can occur year-round in warm climates where the insect is actively feeding. SOLUTION: There is currently no cure. The infected tree must be entirely removed and destroyed to prevent the disease from wiping out neighboring citrus trees."

    # 9. Rots & Molds (Black Rot, Esca, Leaf Mold)
    elif "rot" in label_lower or "esca" in label_lower or "mold" in label_lower:
        return "🟤 WHY IT HAPPENS: Aggressive fungal pathogens that infect the fruit, leaves, or wood, often entering through natural openings or pruning wounds. WHEN IT HAPPENS: Spreads rapidly in warm, extremely damp weather, especially when foliage stays wet overnight. SOLUTION: Prune out dead or diseased areas carefully. Improve sunlight penetration and apply a targeted fungicidal spray early in the growing cycle."

    # 10. Apple Scab
    elif "scab" in label_lower:
        return "🍏 WHY IT HAPPENS: A fungus that overwinters on fallen diseased leaves and shoots spores into the air the following year. WHEN IT HAPPENS: Spreads aggressively during rainy spring weather when new leaves are vulnerable. SOLUTION: Rake and thoroughly destroy fallen leaves in autumn. Apply a preventative fungicide like liquid copper when leaf buds first begin to open."

    # 11. Catch-All for any other issues
    else:
        return "⚠️ WHY IT HAPPENS: Typically caused by fungal or bacterial pathogens thriving in poor environmental conditions. WHEN IT HAPPENS: Often during periods of high humidity, overwatering, or poor airflow. SOLUTION: Isolate the affected plant immediately. Remove diseased foliage, avoid wetting the leaves when watering, and apply a general-purpose organic fungicide."

if __name__ == '__main__':
    # Changed from app.run(debug=True) to Hugging Face production settings
    app.run(host='0.0.0.0', port=7860)