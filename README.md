# Production-Ready RAG Chatbot with Google Gemini API

A modular, production-grade Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Google Gemini API, and Streamlit. This application allows users to upload multiple PDF documents and have conversational interactions with the content using Google's advanced language models.

## Features

- **Multi-PDF Support**: Upload and process multiple PDF documents simultaneously
- **Conversational Memory**: Maintains chat history for contextual conversations
- **Modular Architecture**: Clean separation between UI and RAG logic
- **Production-Ready**: Error handling, configuration management, and robust UI/UX
- **Google Gemini Integration**: Uses latest Gemini models for both embeddings and chat
- **Persistent Vector Store**: ChromaDB for efficient document retrieval

## Technology Stack

- **LLM**: Google Gemini 1.5 Pro Latest
- **Embeddings**: Google Generative AI Embeddings (models/embedding-001)
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **UI**: Streamlit
- **Configuration**: python-dotenv

## Prerequisites

- Python 3.8 or higher
- Google API Key with access to Gemini API
- pip package manager

## Setup Instructions

### 1. Clone or Download the Project

Create a new directory and save all the project files in it.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Google API Key

#### Option A: Environment File (Recommended)
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your Google API Key:
   ```
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

#### Option B: UI Input
If you don't set up the `.env` file, you can enter your API key directly in the Streamlit sidebar when running the application.

### 4. Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated API key
4. Use this key in your `.env` file or enter it in the application UI

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Enter API Key**: If not using a `.env` file, enter your Google API key in the sidebar
2. **Upload PDFs**: Use the file uploader in the sidebar to select one or more PDF files
3. **Wait for Processing**: The application will process your documents and create embeddings
4. **Start Chatting**: Once processing is complete, ask questions about your documents in the main chat interface
5. **Conversational Context**: The chatbot maintains conversation history for contextual responses

## Project Structure

```
/project-root
├── .env.example        # Example environment configuration
├── app.py              # Streamlit UI application
├── rag_pipeline.py     # Core RAG logic and LangChain implementation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Architecture Overview

### Modular Design
- **`rag_pipeline.py`**: Contains all RAG logic, document processing, and chain creation
- **`app.py`**: Handles only UI, state management, and user interactions

### RAG Pipeline Components
1. **Document Loading**: PDF processing using LangChain loaders
2. **Text Splitting**: Intelligent chunking for optimal retrieval
3. **Embeddings**: Google's embedding-001 model for semantic search
4. **Vector Store**: ChromaDB for efficient similarity search
5. **Retrieval Chain**: History-aware retriever with conversational memory
6. **Response Generation**: Google Gemini 1.5 Pro for natural language responses

## Configuration Options

### Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for accessing Gemini services

### Customizable Parameters
You can modify these in `rag_pipeline.py`:
- Chunk size and overlap for text splitting
- Number of retrieved documents
- Temperature and other model parameters

## Error Handling

The application includes comprehensive error handling for:
- Missing or invalid API keys
- PDF processing failures
- Network connectivity issues
- Model API errors

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your Google API key is valid and has Gemini API access
   - Check that the key is correctly set in `.env` or entered in the UI

2. **PDF Processing Errors**
   - Ensure PDFs are not password-protected
   - Try with smaller PDF files if experiencing memory issues

3. **Dependencies Issues**
   - Make sure all requirements are installed: `pip install -r requirements.txt`
   - Use Python 3.8 or higher

4. **Streamlit Issues**
   - Clear browser cache and refresh the page
   - Restart the Streamlit application

### Performance Tips

- For large documents, consider splitting them into smaller files
- The first query after uploading documents may take longer due to embedding creation
- Subsequent queries will be faster as embeddings are cached

## Contributing

This is a production-ready template that can be extended with additional features such as:
- Support for other document formats (Word, TXT, etc.)
- Advanced retrieval strategies
- User authentication
- Document management interface
- API endpoint creation

## License

This project is provided as-is for educational and production use. Please ensure compliance with Google's API terms of service when using Gemini models.

## Support

For issues related to:
- Google API: Check [Google AI documentation](https://ai.google.dev/)
- LangChain: Refer to [LangChain documentation](https://docs.langchain.com/)
- Streamlit: See [Streamlit documentation](https://docs.streamlit.io/)

---

**Note**: This application processes documents locally and sends only relevant chunks to Google's API for response generation. Your full documents are not sent to external services.
