import os  # Imports the 'os' module for interacting with the operating system (e.g., file paths)
import fitz  # Imports the 'fitz' module (PyMuPDF) for working with PDF files (install with: pip install PyMuPDF)
from PIL import Image  # Imports the 'Image' module from Pillow for image processing (install with: pip install Pillow)
import base64  # Imports the 'base64' module for encoding and decoding data in base64 format
from io import BytesIO, StringIO  # Imports the 'BytesIO' and StringIO classes for working with in-memory byte streams
from langchain_google_genai import ChatGoogleGenerativeAI  # Imports the 'ChatGoogleGenerativeAI' class from Langchain for interacting with Gemini (install: pip install langchain-google-genai)
from langchain_core.prompts import ChatPromptTemplate  # Imports 'ChatPromptTemplate' for creating structured prompts
import pandas as pd  # Imports the 'pandas' library for data manipulation and analysis (install with: pip install pandas)
import streamlit as st  # Imports the 'streamlit' library for creating interactive web applications (install with: pip install streamlit)
from dotenv import load_dotenv  # Imports the 'load_dotenv' function for loading environment variables (install with: pip install python-dotenv)
import json  # Imports the 'json' module for working with JSON data
import re  # Imports the 're' module for working with regular expressions (pattern matching in text)
import tempfile  # Imports the 'tempfile' module for creating temporary files and directories
from contextlib import contextmanager  # Imports 'contextmanager' to create context managers for resource management
import csv # Imports the csv module

# Load environment variables from a .env file (if it exists)
load_dotenv()

# Configuration:  Get the Google API Key from environment variables
# Important:  Store your API key securely (e.g., using Streamlit Secrets in Streamlit Cloud)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set a DEBUG flag at the beginning of your script
DEBUG = False  # Set to False to disable debug messages in the Streamlit app. Set to True for debugging.

# Define a class to handle document processing
class DocumentProcessor:
    def __init__(self, api_key=GOOGLE_API_KEY):
        # Constructor: Initializes the DocumentProcessor with the Google API key
        if not api_key:
            # If the API key is missing, raise a ValueError to prevent the app from running without it
            raise ValueError("GOOGLE_API_KEY not found in environment variables.  Please set it in a .env file or Streamlit Secrets.")
        self.gemini_vision = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Specifies the Gemini model to use
            google_api_key=api_key,  # Passes the Google API key
            convert_system_message_to_human=True #This argument in `ChatGoogleGenerativeAI` makes sure that system messages are converted into human messages. It is required as otherwise the prompt is not valid
        ) # Initializes the Gemini Vision API client

    def _image_to_base64(self, image):
        # Helper function: Converts a PIL Image object to a base64 encoded string
        buffered = BytesIO()  # Creates an in-memory byte stream
        if image.mode == 'RGBA':
            # Convert RGBA to RGB if needed
            image = image.convert('RGB')
        image.save(buffered, format="JPEG")  # Saves the image to the byte stream in JPEG format
        return base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encodes the byte stream to base64 and returns the string

    def process_pdf(self, file_path):
        # Function: Extracts images from a PDF file
        doc = fitz.open(file_path)  # Opens the PDF file using PyMuPDF
        images = []  # Initializes an empty list to store the images
        for page in doc:  # Iterates through each page in the PDF
            pix = page.get_pixmap()  # Renders the page as a pixmap (image)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Creates a PIL Image object from the pixmap data
            images.append(img)  # Appends the image to the list
        return images  # Returns the list of images

    def extract_data_with_gemini(self, image):
        # Function: Extracts data from an image using the Gemini Vision API
        try:
            base64_image = self._image_to_base64(image)  # Converts the image to a base64 string

            # Define the prompt for the Gemini Vision API
            # UPDATED PROMPT!
            prompt = ChatPromptTemplate.from_messages([
                ("human", [
                    {"type": "text", "text": """Extract the lists of names of companies, organizations, and entities from this image.
                                                 The image contains multiple columns of names.  Identify each column and extract the names from each column separately.
                                                 Format the output as a CSV file with a header row indicating the column number ("Column 1", "Column 2", etc.).
                                                 The CSV data MUST have a consistent number of columns in each row.  If any row has missing values, fill them with an empty string.
                                                 Return ONLY the CSV data. Do not include any introductory or explanatory text."""}, # The prompt tells the model what to do.
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}  # Includes the base64 encoded image in the prompt
                ])
            ])

            messages = prompt.format_messages() # Format the message to be sent to gemini
            response = self.gemini_vision.invoke(messages) # Invoke the gemini API and get response
            content = response.content # Get the content from the API response

            return content # now we need CSV format
        except Exception as e:
            # If any other error occurs during the process, log the error and return None
            st.error(f"Gemini processing error: {str(e)}")
            return None

    def convert_to_excel_format(self, csv_data):
        # Function: Converts CSV data to Pandas DataFrame, handles inconsistent rows
        if csv_data is None:
            st.error("No CSV data to convert.")
            return None

        try:
            csv_buffer = StringIO(csv_data)
            reader = csv.reader(csv_buffer)

            # Read all rows into a list
            rows = list(reader)

            # Determine the maximum number of columns
            max_columns = max(len(row) for row in rows) if rows else 0 # handles case where the csv_data is empty

            # Pad rows with missing values
            for row in rows:
                while len(row) < max_columns:
                    row.append("")

            # Create DataFrame from padded rows
            df = pd.DataFrame(rows)

            return df # return dataframe for display.

        except Exception as e:
            st.error(f"CSV conversion error: {str(e)}")
            return None



# Function to safely handle temporary files using context manager
@contextmanager
def create_temporary_file(uploaded_file):
    """
    Creates a temporary file from an uploaded file object and returns its path.
    The temporary file is automatically deleted when the context manager exits.
    """
    try:
        suffix = os.path.splitext(uploaded_file.name)[1] # Get the suffix (extension) of the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file: # create temporary file
            tmp_file.write(uploaded_file.read())  # Writes the uploaded file content to the temporary file
            tmp_file_path = tmp_file.name  # Get the name of the temporary file
        yield tmp_file_path  # Yield the temporary file path to the context

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path) # Delete the temporary file


# Streamlit UI
def main():
    # Function:  Defines the main function for the Streamlit application
    st.set_page_config(page_title="Doc2Excel Pro", page_icon="📑")  # Sets the page title and icon
    st.title("📑 Nishu App-Powered Document Converter")  # Sets the title of the Streamlit app

    try:
        processor = DocumentProcessor()  # Initializes the DocumentProcessor
    except ValueError as e:
        # If the API key is not found, display an error message and exit
        st.error(str(e))
        return

    # File uploader widget: Allows users to upload files
    uploaded_file = st.file_uploader("PDF/Image अपलोड करें", type=["pdf", "png", "jpg", "jpeg"]) # Set the file types accepted

    if uploaded_file:
        # If a file has been uploaded, proceed with processing
        with st.status("Processing...", expanded=True) as status:
            # Displays a status message while processing
            try:
                # Step 1: Process file
                if uploaded_file.type == "application/pdf":
                    # If the uploaded file is a PDF, process it using the following code
                    # Use the context manager for temporary file handling
                    with create_temporary_file(uploaded_file) as temp_pdf_path:
                        try:
                            images = processor.process_pdf(temp_pdf_path)  # Pass the temporary file path
                        except Exception as e:  # Catch more general exceptions
                            st.error(f"Error processing PDF: {e}")
                            return

                    all_data = []
                    for img in images:
                        csv_data = processor.extract_data_with_gemini(img) # this returns csv
                        if csv_data:
                            df = processor.convert_to_excel_format(csv_data) # this returns pandas dataframe
                            if df is not None:
                                all_data.append(df)

                    if not all_data:
                        st.error("Could not extract any valid data from the PDF.")
                        return

                    df = pd.concat(all_data, axis = 1) #Concatenate dataframes column wise.

                else: # Image processing
                    # If the uploaded file is an image, process it using the following code
                    img = Image.open(uploaded_file)
                    csv_data = processor.extract_data_with_gemini(img)  # this returns csv
                    if csv_data:
                        df = processor.convert_to_excel_format(csv_data)  # this returns pandas dataframe
                        if df is None:
                            st.error("Could not convert to CSV.")
                            return
                    else:
                        st.error("Could not extract JSON data from the image.")
                        return

                # Step 3: Show results
                status.update(label="Processing Complete!", state="complete") # Update the status message
                st.dataframe(df) # Show the Dataframe on Streamlit app

                # Download button
                excel_file = BytesIO()
                df.to_csv(excel_file, index=False)

                st.download_button(
                    label="Download CSV",
                    data=excel_file.getvalue(),
                    file_name="converted_data.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Main processing error: {str(e)}") # If an exception occurs, show the error message

if __name__ == "__main__":
    # Executes the `main` function when the script is run
    main()