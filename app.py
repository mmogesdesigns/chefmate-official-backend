from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv 
from flask_cors import CORS, cross_origin
import base64
import google.generativeai as genai

load_dotenv() # Load environment variables from a .env file
app = Flask(__name__) 
CORS(app, resources={r"/*": {"origins": "https://chefmate.netlify.app/"}}) # Allow CORS for the frontend
RENDER_API_URL = os.getenv('RENDER_API_URL');

app.secret_key = 'secret-key' 

EDAMAM_API_ID = os.getenv('EDAMAM_API_ID') # Get the Edamam API ID from the .env file
EDAMAM_API_KEY = os.getenv('EDAMAM_API_KEY')
api_key = os.getenv("GEMINI_API_KEY") # Get the Gemini API key from the .env file

if not api_key: # Check if the API key is set
    print("Error: GEMINI_API_KEY is not set in the environment variables.")
genai.configure(api_key=api_key) # Configure the API client with the API key

def upload_to_gemini(base64_image, mime_type="image/jpeg"): 
    """Uploads the given file to Gemini.""" 
    try: 
        file = genai.upload_file(base64_image, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        print(f"Error uploading file to Gemini: {e}")
        raise e

generation_config = { # Configuration for content generation
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel( 
    model_name="gemini-1.5-pro", 
    generation_config=generation_config, 
)

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    try:
        data = request.get_json() # Get JSON data from the request
        image_base64 = data['imageBase64'] # Get the base64 encoded image from the data
        print(f"Received image base64 length: {len(image_base64)}") # Log the length of the image data

        # Save the image to a temporary file
        image_data = base64.b64decode(image_base64)
        temp_file_path = "temp_image.jpeg"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(image_data)

        # Upload the file to Gemini
        uploaded_file = upload_to_gemini(temp_file_path)

        # Generate content using the model
        prompt = (
            "Extract and list all the food items from the provided image."
            "Provide the names in alphabetical order. "
            "Don't include non-food items in the list. List each item on a new line."
        )
        response = model.generate_content([
            prompt,
            "Image: ",
            uploaded_file,
            "List of Objects: "
        ])

        # Extract the detected objects from the response and remove duplicates
        detected_objects = response.text.split("\n")
        detected_objects = [obj.strip() for obj in detected_objects if obj.strip()]
        detected_objects = list(set(detected_objects))  # Remove duplicates
        detected_objects.sort()  # Sort alphabetically
        print(f"Detected objects: {detected_objects}")

        # Remove the temporary file
        os.remove(temp_file_path)

        return jsonify({"detectedObjects": detected_objects}) # Return the detected objects as JSON
    except Exception as e:
        print(f"Error during object detection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
@cross_origin(origins="https://chefmate.netlify.app")
def search_recipes(): # Function to search for recipes
    data = request.get_json() 
    ingredient = data.get('recipe_name') # Get the ingredient name from the JSON data
    print(f"Received ingredient: {ingredient}")  # Logging the received data
    if not ingredient: 
        return jsonify({'error': 'Please enter an ingredient name.'}), 400

    results = get_recipes(ingredient, 0, 24) # Get recipes for the given ingredient
    if results: # Check if recipes were found
        return jsonify({'results': results})
    else: # Return an error if no recipes were found
        return jsonify({'error': f"No recipes found for '{ingredient}'."}), 404

def get_recipes(query, start=0, end=24): # Function to fetch recipes from the Edamam API
   # Get the Edamam API key from the .env file
    url = f"https://api.edamam.com/search?q={query}&app_id={EDAMAM_API_ID}&app_key={EDAMAM_API_KEY}&from={start}&to={end}" # Create the API URL
    try:
        response = requests.get(url) # Send a GET request to the Edamam API
        response.raise_for_status() 
        data = response.json()
        return data.get('hits', []) 
    except requests.exceptions.RequestException as e: 
        print(f"Failed to fetch recipes for '{query}': {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True)