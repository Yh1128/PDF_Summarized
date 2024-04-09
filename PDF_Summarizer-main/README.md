# PDF Summarizer

Summarize text using **ChatGPT**

## **🚀 How to Use**

1. Ensure that you have the necessary dependencies installed, including PyPDF2, Streamlit, Langchain, and more. You can install them using pip:

    ```bash
    pip install PyPDF2 streamlit langchain
    ```

2. Create an OpenAI API key by signing up on the OpenAI platform.

3. Create a `.env` file with your OpenAI API key; you can name it `openai.env` and store it in the same directory as your script.

4. Run the script using the following command:

    ```bash
    streamlit run app.py
    ```

5. In the Streamlit web interface, click the "Upload your PDF Document" button to upload your PDF file.

6. The script will extract the text from the PDF and generate a summary of the content.

## **🚀 Dependencies**

- Python (>= 3.6)
- PyPDF2: Used to extract text from PDF documents.
- Streamlit: Creates a user-friendly web interface.
- Langchain: Provides text processing and summarization capabilities.
- Hugging Face Transformers: Utilized for text embeddings.
- FAISS: A library for efficient similarity search.
- OpenAI GPT-3.5 Turbo: Generates text-based responses.

## **🚀 Functionality**

1. The script extracts text from the uploaded PDF document.

2. The extracted text is divided into chunks to create a knowledge base.

3. A user-provided query is used to search the knowledge base for relevant information.

4. OpenAI's GPT-3.5 Turbo is used to generate a summary response based on the query and the knowledge base.

5. The summary results are displayed in the Streamlit interface.

## **🚀 Limitations**

- The quality of the summary depends on the content and structure of the uploaded PDF document.

- Longer documents may not be summarized effectively due to API limits and chunking.

- The summarization may not capture highly technical or specific information.

Enjoy 😄 summarizing your PDF documents quickly and efficiently with this Python script!
