
# PDF Table Extraction and YOLO Object Detection

This project processes PDF files by converting them into images, performing YOLO object detection on the images, and then extracting table data from detected objects using the Google Gemini API. The results are saved in a JSON file.

## Features

- **PDF to Image Conversion:** Converts each page of a PDF into images.
- **YOLO Object Detection:** Detects objects in the images using a pre-trained YOLO model.
- **Google Gemini API Integration:** Extracts table data from detected objects using the Gemini API.
- **Asynchronous Processing:** Utilizes asynchronous programming to efficiently handle multiple PDF files and API requests.
- **Auto-Saving of Results:** Saves extracted results into a JSON file after processing every 5 images.

## Prerequisites

- Python 3.8+
- [YOLOv8](https://github.com/ultralytics/yolov8)
- Google Gemini API
- Required Python packages (see below)

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/pdf-table-extraction.git
    cd pdf-table-extraction
    ```

2. **Install Required Python Packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**

   Create a `.env` file in the root directory with the following variables:

    ```env
    GEMINI_API_KEY=your_google_gemini_api_key
    PDF_FOLDER_PATH=path_to_your_pdf_folder
    ```

## Usage

1. **Prepare Your PDFs:**
   
   Place all the PDF files you want to process in the folder specified in the `PDF_FOLDER_PATH` environment variable.

2. **Run the Script:**

    ```bash
    python your_script_name.py
    ```

3. **View the Results:**

   The results will be saved in a JSON file named `gemini_results.json` in the current working directory. This file will contain the extracted table data from your PDFs.

## Code Structure

- **Main Processing Functions:**
  - `get_pdf_files`: Scans the specified folder for PDF files.
  - `convert_pdf_to_images_async`: Converts each page of a PDF into an image asynchronously.
  - `perform_yolo_predictions`: Performs YOLO object detection on a list of images.
  - `extract_table_from_image_async`: Extracts table data from images using the Google Gemini API.
  - `save_results_to_json_async`: Saves the results to a JSON file asynchronously.

- **Helper Functions:**
  - `send_request_with_retry_async`: Sends an asynchronous request to the Gemini API with retry logic.
  - `clean_json_string`: Cleans a JSON string by removing unnecessary characters.
  - `parse_gemini_response`: Parses the response from Gemini API that contains concatenated JSON objects.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
