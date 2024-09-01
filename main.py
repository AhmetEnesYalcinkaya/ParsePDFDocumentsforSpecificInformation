import os
import cv2
import time
import re
import json
import concurrent.futures
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv
from colorama import Fore, init
from pdf2image import convert_from_path
from ultralytics import YOLO
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.api_core import exceptions

# Suppress ultralytics logging output
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Initialize colorama for colored terminal output
init(autoreset=True)

# Load environment variables from a .env file
load_dotenv()

# Configure the Gemini API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the YOLO model
model = YOLO("best.pt")

# Find PDF files in the specified directory
def get_pdf_files(pdf_folder_path):
    """
    Scans the specified folder for PDF files.

    :param pdf_folder_path: Path to the folder containing PDF files.
    :return: List of PDF filenames.
    """
    print(f"{Fore.CYAN}Scanning for PDF files in {pdf_folder_path}...")
    return [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# Convert PDF to images asynchronously
async def convert_pdf_to_images_async(pdf_path, output_folder):
    """
    Converts each page of a PDF into an image asynchronously.

    :param pdf_path: Path to the PDF file.
    :param output_folder: Folder to save the resulting images.
    :return: List of paths to the converted images.
    """
    pages = await asyncio.to_thread(convert_from_path, pdf_path, 100)
    print(f"{Fore.BLUE}Converted {os.path.basename(pdf_path)} to images. Total pages: {len(pages)}")
    
    image_paths = []
    for i, page in enumerate(pages):
        image_name = f"page_{i+1}.jpg"
        image_path = os.path.join(output_folder, image_name)
        await asyncio.to_thread(page.save, image_path, 'JPEG')
        image_paths.append(image_path)
    
    return image_paths

# Perform YOLO predictions and save results (using thread pool)
def perform_yolo_predictions(image_paths, predicted_folder, extracted_folder):
    """
    Performs YOLO object detection on a list of images and saves the results.

    :param image_paths: List of image file paths to perform predictions on.
    :param predicted_folder: Folder to save images with predicted boxes.
    :param extracted_folder: Folder to save cropped images of detected objects.
    :return: List of results including the predictions.
    """
    all_results = []
    print(f"{Fore.CYAN}Starting YOLO predictions on images...")

    for image_path in image_paths:
        try:
            result = process_single_image(image_path, predicted_folder, extracted_folder)
            all_results.extend(result)
        except Exception as exc:
            print(f'{image_path} generated an exception: {exc}')
    
    print(f"{Fore.CYAN}Completed YOLO predictions on all images.")
    return all_results

def process_single_image(image_path, predicted_folder, extracted_folder):
    """
    Processes a single image for YOLO predictions, saves the detected objects, and the annotated image.

    :param image_path: Path to the image file to process.
    :param predicted_folder: Folder to save the image with predicted boxes.
    :param extracted_folder: Folder to save cropped images of detected objects.
    :return: List of results including the box coordinates, cropped image path, and confidence scores.
    """
    results = model.predict(source=image_path, verbose=False, conf=0.6)
    
    img = cv2.imread(image_path)
    image_results = []
    for j, result in enumerate(results):
        for k, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = box.conf[0].item()  # Get the confidence score of the prediction
            
            # Draw a rectangle around the detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Put the confidence score on the image
            label = f'Score: {score:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_name = f"{os.path.basename(image_path)[:-4]}_box_{j+1}_{k+1}.jpg"
            cropped_img_path = os.path.join(extracted_folder, cropped_img_name)
            cv2.imwrite(cropped_img_path, cropped_img)

            predicted_img_name = f"{os.path.basename(image_path)[:-4]}_predicted.jpg"
            predicted_img_path = os.path.join(predicted_folder, predicted_img_name)
            cv2.imwrite(predicted_img_path, img)
            
            image_results.append({
                "image_path": image_path,
                "box": k + 1,
                "extracted_image_name": cropped_img_name,
                "cropped_img_path": cropped_img_path,
                "score": score  # Include the score in the results
            })

    print(f"{Fore.YELLOW}Processed {image_path} for YOLO predictions.")
    
    os.remove(image_path)
    
    return image_results

# Send asynchronous request to Gemini API with retry logic
async def send_request_with_retry_async(session, message, max_retries=10):
    """
    Sends an asynchronous request to the Gemini API with retry logic.

    :param session: The session object for the Gemini API.
    :param message: The message to send to the Gemini API.
    :param max_retries: Maximum number of retry attempts.
    :return: The response from the Gemini API.
    """
    for attempt in range(max_retries):
        try:
            response = await session.send_message_async(message)
            return response
        except exceptions.ResourceExhausted:
            wait_time = 15 * (2 ** attempt)  # Exponential backoff
            print(f"API quota exceeded, waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait_time)
        except exceptions.InternalServerError:
            wait_time = 20 * (2 ** attempt)  # Exponential backoff
            print(f"Internal server error, waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait_time)
    raise Exception("Max retries exceeded, resource still exhausted or internal server error persists.")

# Upload file to Google Gemini API
def upload_to_gemini(path, mime_type=None):
    """
    Uploads the given file to the Gemini API.

    :param path: Path to the file to be uploaded.
    :param mime_type: MIME type of the file being uploaded (optional).
    :return: The uploaded file's metadata including URI.
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    
    file = genai.upload_file(path, mime_type=mime_type)
    if not file or not file.uri:
        raise ValueError(f"Failed to upload file: {path}")
    
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Function to clean JSON strings before parsing
def clean_json_string(json_string):
    """
    Cleans a JSON string by removing unnecessary characters.

    :param json_string: The raw JSON string to clean.
    :return: The cleaned JSON string.
    """
    # + operatörünü ve boşlukları kaldırarak stringleri birleştirir
    cleaned_string = re.sub(r'\s*\+\s*', '', json_string)
    return cleaned_string

# Function to parse Gemini response
def parse_gemini_response(response_text):
    """
    Parses the response from Gemini API that contains concatenated JSON objects.

    :param response_text: The raw text response from the Gemini API.
    :return: A list of parsed JSON objects.
    """
    # Regex to match valid JSON objects within the string
    json_objects = re.findall(r'\{.*?\}', response_text)

    parsed_responses = []
    for obj in json_objects:
        cleaned_obj = clean_json_string(obj)  # Clean the JSON string
        try:
            parsed_obj = json.loads(cleaned_obj)
            parsed_responses.append(parsed_obj)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON object: {cleaned_obj}")
    
    return parsed_responses

# Extract table from image using Gemini API
async def extract_table_from_image_async(gemini_model, cropped_img_path):
    """
    Extracts a table from an image using the Gemini API.

    :param gemini_model: The Gemini model instance.
    :param cropped_img_path: Path to the cropped image containing the table.
    :return: Parsed response containing table data in JSON format.
    """
    # Upload the image to Gemini
    uploaded_file = await asyncio.to_thread(upload_to_gemini, cropped_img_path, mime_type="image/jpeg")
    if not uploaded_file:
        raise ValueError(f"Uploaded file is empty or invalid: {cropped_img_path}")
    
    print(f"{Fore.CYAN}File uploaded successfully. Preparing to start chat session with Gemini API...")
    
    # Define the message to extract the table
    message = ("""
                There should be a table in the photo. If there is a table,
                Parse the table's columns ("MANUFACTURER", "MODEL" and "SERVICE LOCATION/REGION" ) in this image into JSON format.
                If there is no information about these columns, leave them blank and do not enter anything.
                
            """)
    # Start a chat session with Gemini API
    chat_session = await asyncio.to_thread(gemini_model.start_chat, history=[
        {
            "role": "user",
            "parts": [
                uploaded_file,
            ],
        },
    ])
    
    print(f"{Fore.CYAN}Chat session started. Sending request to extract table from the image...")
    
    # Send the request with retry logic
    response = await send_request_with_retry_async(chat_session, message)
    
    if not response or not response.text:
        raise ValueError("The response from LLM is empty or invalid.")
    
    print(f"{Fore.CYAN}Received response from Gemini API.")
    
    # Parse the response and return it
    parsed_response = parse_gemini_response(response.text)
    return parsed_response

# Process all PDFs asynchronously
async def process_pdfs_async(pdf_folder_path, output_folder, gemini_model):
    """
    Processes all PDF files in a folder asynchronously by converting them to images, 
    performing object detection, and extracting tables from detected objects.

    :param pdf_folder_path: Path to the folder containing PDF files.
    :param output_folder: Folder to save the processed images.
    :param gemini_model: The Gemini model instance for table extraction.
    :return: List of results including the PDF filename, extracted image name, and Gemini response.
    """
    print(f"{Fore.CYAN}Starting PDF processing in {pdf_folder_path}...")
    pdf_files = get_pdf_files(pdf_folder_path)
    all_results = []
    
    save_interval = 5  # Save JSON after every 5 images
    output_json_path = os.path.join(os.getcwd(), "gemini_results.json")
    
    for pdf_file in pdf_files:
        pdf_name = pdf_file[:-4]
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        pdf_images_folder = os.path.join(output_folder, pdf_name)
        predicted_folder = os.path.join(pdf_images_folder, 'predicted_table')
        extracted_folder = os.path.join(pdf_images_folder, 'extracted_tables')

        os.makedirs(predicted_folder, exist_ok=True)
        os.makedirs(extracted_folder, exist_ok=True)

        print(f"{Fore.CYAN}Processing {pdf_file}...")
        image_paths = await convert_pdf_to_images_async(pdf_path, pdf_images_folder)
        yolo_results = await asyncio.to_thread(perform_yolo_predictions, image_paths, predicted_folder, extracted_folder)

        for result in yolo_results:
            response_text = await extract_table_from_image_async(gemini_model, result["cropped_img_path"])
            for parsed_response in response_text:
                all_results.append({
                    "pdf_file": pdf_file,
                    "extracted_image_name": result["extracted_image_name"],
                    "gemini_response": parsed_response
                })

            print(f"{Fore.GREEN}Processed and received response for {result['extracted_image_name']} from {pdf_file}.")
            await asyncio.sleep(2)  # 2-second delay between API calls
            
            # Save results every 5 images
            if len(all_results) % save_interval == 0:
                await save_results_to_json_async(all_results, output_json_path)
                all_results.clear()  # Clear the list after saving to avoid duplicate entries

        print(f"{Fore.GREEN}Completed processing for {pdf_file}.")
    
    # Save any remaining results that were not saved in the last interval
    if len(all_results) % save_interval != 0:
        await save_results_to_json_async(all_results, output_json_path)
        all_results.clear()  # Clear the list after saving the remaining results
    
    print(f"{Fore.YELLOW}All PDFs processed successfully and results saved to {output_json_path}.")
    return all_results

# Save the results to a JSON file asynchronously
async def save_results_to_json_async(results, output_json_path):
    """
    Saves the results to a JSON file asynchronously, appending if the file already exists.

    :param results: List of results to save.
    :param output_json_path: Path to the JSON file where results will be saved.
    """
    formatted_results = []
    
    for result in results:
        try:
            gemini_response = result.get("gemini_response")
            
            # Check if gemini_response is not None and is a dictionary
            if gemini_response and isinstance(gemini_response, dict):
                manufacturers = gemini_response.get("MANUFACTURER", [])
                models = gemini_response.get("MODEL", [])
                locations = gemini_response.get("LOCATION/AREA SERVED", [])
                
                # Ensure these fields are lists
                if isinstance(manufacturers, str):
                    manufacturers = [manufacturers]
                if isinstance(models, str):
                    models = [models]
                if isinstance(locations, str):
                    locations = [locations]

                if not manufacturers and not models:
                    locations = []

                max_length = max(len(manufacturers), len(models), len(locations))
                manufacturers = manufacturers + [''] * (max_length - len(manufacturers))
                models = models + [''] * (max_length - len(models))
                locations = locations + [''] * (max_length - len(locations))
                
                for manufacturer, model, location in zip(manufacturers, models, locations):
                    formatted_result = {
                        "pdf_file": result["pdf_file"],
                        "extracted_image_name": result["extracted_image_name"],
                        "MANUFACTURER": manufacturer,
                        "MODEL": model,
                        "LOCATION/AREA SERVED": location
                    }
                    formatted_results.append(formatted_result)
            else:
                print(f"{Fore.RED}gemini_response is None or not a dictionary.")
        except Exception as e:
            print(f"{Fore.RED}Error processing response: {e}")
    
    # Save or append the results to the JSON file
    if os.path.exists(output_json_path):
        if os.path.getsize(output_json_path) > 0:  # Check if file is not empty
            with open(output_json_path, "r+", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_data.extend(formatted_results)
                f.seek(0)
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(formatted_results, f, ensure_ascii=False, indent=4)
    else:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=4)
    
    print(f"{Fore.YELLOW}Results saved to {output_json_path}.")

# Main processing function (asynchronous)
async def main_async():
    """
    The main function that initializes the processing of PDFs and calls all other functions.
    """
    pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
    output_folder = os.path.join(os.getcwd(), 'images')

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            required=["MANUFACTURER", "MODEL", "LOCATION/AREA SERVED"],
            properties={
                "MANUFACTURER": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(type=content.Type.STRING),
                ),
                "MODEL": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(type=content.Type.STRING),
                ),
                "LOCATION/AREA SERVED": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(type=content.Type.STRING),
                ),
            },
        ),
        "response_mime_type": "application/json",
    }

    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    results = await process_pdfs_async(pdf_folder_path, output_folder, gemini_model)

if __name__ == "__main__":
    asyncio.run(main_async())
