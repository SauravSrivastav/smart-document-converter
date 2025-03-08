# Smart Document Converter

## Description

The Smart Document Converter is an AI-powered tool that extracts data from images and PDFs using Google's Gemini Vision API and converts it to CSV format. It leverages Streamlit for a user-friendly interface, allowing users to upload documents and download the extracted data.

## Features

*   **Document Upload:** Supports PDF and image (PNG, JPG, JPEG) files.
*   **AI-Powered Extraction:** Uses Gemini Vision API to intelligently extract data, even from complex documents.
*   **CSV Export:** Converts extracted data into a CSV file for easy use in spreadsheet applications.
*   **Streamlit Interface:** Provides a simple and intuitive user experience.

## How to Use

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:YOUR_GITHUB_USERNAME/smart-document-converter.git
    cd smart-document-converter
    ```

    Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

2.  **Create a `.env` file:**

    Create a `.env` file in the project directory with the following content:

    ```
    # Note: DO NOT include your actual API Key directly in the .env file.
    # This is handled via Streamlit Secrets.
    ```

    (This is because we're using Streamlit Secrets, so no need to include the key here.)

3.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
*   Run `pip freeze > requirements.txt` to create a file called `requirements.txt` with all dependencies so that Streamlit Cloud can install the packages when deploying

5.  **Set up Streamlit Secrets:**

    *   Deploy the app to Streamlit Cloud: [https://streamlit.io/cloud](https://streamlit.io/cloud)
    *   In your Streamlit app's dashboard, go to the "Secrets" section.
    *   Create a secret named `GOOGLE_API_KEY` and paste your API key as the value.

6.  **Run the Streamlit app locally (optional):**

    ```bash
    streamlit run app.py
    ```

7.  **Access the app:**

    Open your web browser and go to the address displayed in the Streamlit console (usually `http://localhost:8501`).

## Deployment

The app is deployed on Streamlit Cloud and can be accessed here: [YOUR_STREAMLIT_APP_URL] (Replace with your actual URL after deployment).

## Contributing

Feel free to contribute to the project by submitting issues or pull requests.

## Important Notes

*   Remember to keep your Google API key secure. Store it using Streamlit Secrets or a similar secure method.
*   The accuracy of data extraction depends on the quality of the input image and the capabilities of the Gemini Vision API.  Experiment with different prompts if needed.
