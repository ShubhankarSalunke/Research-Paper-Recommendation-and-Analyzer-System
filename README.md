#Research Paper Recommender and Analyzer

# Research Paper Recommender and Analyzer

## Description
The Research Paper Recommender and Analyzer is a web application designed to help users find and analyze research papers. It utilizes advanced Natural Language Processing (NLP) techniques and machine learning algorithms to provide recommendations based on user queries. Users can search for relevant papers, compare different papers, and analyze the content of uploaded PDF documents.

## Features
- **Search for Research Papers**: Users can input queries to find relevant research papers in various domains.
- **Paper Comparison**: Select multiple papers to compare their contents and similarities.
- **Content Analysis**: Upload PDF research papers for content extraction and analysis, including detailed summaries.
- **Language Support**: Supports multiple languages for search queries and translations.
- **Sci-Hub Availability Check**: Check the availability of research papers on Sci-Hub.

## Technologies Used
- **Python**: Main programming language.
- **Streamlit**: Framework for building the web application.
- **Pandas**: Data manipulation and analysis.
- **spaCy**: NLP library for text processing.
- **sklearn**: For machine learning and cosine similarity calculations.
- **Google Generative AI**: For query optimization and text summarization.
- **Beautiful Soup**: For web scraping.
- **Requests**: For making HTTP requests.
- **Google Translate API**: For language translation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/research-paper-recommender.git
   cd research-paper-recommender
2. Install the required packages:
   pip install -r requirements.txt
3. Set up your Google API key:
    Sign up for Google Cloud and create a project.
    Enable the Generative AI and Translation APIs.
    Add your API key to the environment variables or directly in the code (not recommended for production).
4. Run the application:
    ```
    streamlit run app.py
## Usage

    Visit the web application in your browser (usually at http://localhost:8501).
    Use the navigation menu to search for research papers, analyze uploaded PDFs, or compare papers.
    Follow the prompts to input queries and upload files.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.
