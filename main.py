import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import fitz
import google.generativeai as genai
import re
import io
import os
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate
import requests
from bs4 import BeautifulSoup
import urllib.request
from googletrans import Translator

global all_papers
translator = Translator()

languages = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh-cn",
    "Hindi": "hi",
    "Japanese": "ja"
} 

def compare_papers(selected_titles, all_papers):
    if not selected_titles:
        st.write("No papers selected for comparison.")
        return

    # Create an empty list to store matched rows
    matched_rows = []

    # Loop through each selected paper title
    for paper_title in selected_titles:
        # Fetch the row from all_papers that matches the paper title
        row = all_papers[all_papers['Title'] == paper_title]
        if not row.empty:
            matched_rows.append(row)

    if matched_rows:
        result_df = pd.concat(matched_rows).reset_index(drop=True)  # Reset index here
        return result_df
    else:
        st.write("No matching papers found.")
        return None

def translate_to_english(query):
    translator = Translator()
    try:
        translated = translator.translate(query, dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None


nlp = spacy.load("en_core_web_sm")

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.1,
    google_api_key="AIzaSyBpVRT86uMPCk7tKX_q-x3Ula8U8ucaiMA",
)


def check_scihub_availability(doi):
    sci_hub_url = "https://sci-hub.se/"
    check_url = sci_hub_url + doi
    try:
        response = requests.get(check_url, timeout=10)
        if response.status_code == 200:
            if "Sci-Hub: article not found" in str(response.content):
                flag = False
            else:
                flag = True
        else:
            print(
                f"Could not access Sci-Hub for DOI {doi}. Status code: {response.status_code}"
            )
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
    return flag, check_url


def get_pdf_src(url):
    response = requests.get(url)
    html_content = response.text

    # Parse and extract src as shown above
    soup = BeautifulSoup(html_content, "html.parser")
    embed_tag = soup.find("embed", {"type": "application/pdf"})

    if embed_tag and "src" in embed_tag.attrs:
        pdf_src = embed_tag["src"]
        return pdf_src


import urllib.request


def download_pdf_with_urllib(url):
    pdf_filename = url.split("/")[-1].split("#")[0]
    try:
        urllib.request.urlretrieve(url, pdf_filename)
        print(f"PDF downloaded successfully: {pdf_filename}")
    except Exception as e:
        print(f"Failed to download PDF: {e}")

    return pdf_filename


def optimize_query(input_text):
    prompt2 = ChatPromptTemplate(
        input_variables=["query"],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    template="You are a helpful assistant that improves queries for researching papers by correcting grammar and making them more clear and elaborate."
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template="Improve the grammar and clarity of this query with more elaboration (maximum 50 words): {query}\nOutput only the improved query:"
                )
            ),
        ],
    )
    output_query = prompt2 | gemini | StrOutputParser()
    input_text = output_query.invoke(input_text)

    return input_text


def transform_text_spacy(text):
    text = text.lower()
    doc = nlp(text)

    tokens = []

    for token in doc:
        if (
            token.is_alpha
            and token.text not in STOP_WORDS
            and token.text not in string.punctuation
        ):
            tokens.append(token.lemma_)

    return " ".join(tokens)




def calculate_cosine_similarity(query, data):
    data["processed_text"] = data["Title"] + " " + data["Abstract_Spacy"]
    data["processed_text"] = data["processed_text"].apply(transform_text_spacy)
    query = transform_text_spacy(query)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["processed_text"])

    query_vec = vectorizer.transform([query])

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    return cosine_similarities


def search_papers(query, data, top_n=5):
    query = optimize_query(query)
    st.write("Optimized Query: ")
    st.write(query)
    similarities = calculate_cosine_similarity(query, data)
    data["similarity"] = similarities

    # Sort papers by similarity score
    sorted_data = data.sort_values(by="similarity", ascending=False).head(top_n)

    # Extract relevant information
    results = []
    for _, row in sorted_data.iterrows():
        paper_info = {
            "Title": row["Title"],
            "Abstract": row["Cleaned_Abstract"],
            "Link": row["Link"],
            "Journal": row["Journal"],
            "Citations": row["Citations"],
            "Publication Date": row["Publication Date"],
            "Authors": row["Authors"],
        }
        results.append(paper_info)

    return results


# Set page configuration
st.set_page_config(
    page_title="Research Paper Recommender", page_icon="🔍", layout="centered"
)


# Apply CSS for minimalistic design
st.markdown(
    """
    <style>
    .search-box {
        width: 60%;
        margin: auto;
        padding: 20px;
        text-align: center;
    }
    .search-button {
        display: block;
        margin: 20px auto;
        padding: 10px 20px;
        background-color: #4285F4;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .search-button:hover {
        background-color: #357ae8;
    }
    .card {
        margin: 10px 0;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .card-title {
        font-size: 18px;
        font-weight: bold;
    }
    .card-description {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation bar using streamlit-option-menu
selected = option_menu(
    menu_title=None,  # Required
    options=["Home", "Search", "Analyze", "Compare" ,"About"],  # Menu options
    icons=["house", "search", "analyze", "compare", "info-circle"],  # Icons for each menu option
    menu_icon="cast",  # Optional icon for the menu
    default_index=0,  # Which option is selected by default
    orientation="horizontal",  # Horizontal menu
)

# Title of the page
st.markdown(
    "<h1 style='text-align: center; font-family: sans-serif;'>Research Paper Recommender and Analyzer</h1>",
    unsafe_allow_html=True,
)

# Home Page
if selected == "Home":
    st.write(
        "Welcome to the Research Paper Recommender and Analyzer. Use the navigation menu to search for papers or learn more about the app."
    )

# Search Papers Page
elif selected == "Search":


    language_selection = st.selectbox("Select your preferred language:", list(languages.keys()))
    selected_language = languages[language_selection]
    
    paper_types = st.multiselect(
        "Select domain:",
        [
            "Artificial Intelligence",
            "Blockchain and Cryptocurrency",
            "Cloud Computing",
            "Cyber Security",
            "Data Science and Big Data",
            "Human Computer Interaction",
            "IOT",
            "Quantum Computing",
            "Robotics and Automation",
            "Software Engineering",
        ],
    )
    if paper_types:
        dfs = []
        for name in paper_types:
            df = pd.read_csv(rf"data\Cleaned_Data_Improved\{name}.csv")
            dfs.append(df)
        combined_data = pd.concat(dfs, ignore_index=True)
        combined_data = combined_data.dropna()

    with st.form(key="search_form"):
        query = st.text_input(
            "Search relevant research papers",
            placeholder="Type something...",
            label_visibility="collapsed",
        )
        search_button = st.form_submit_button(label="🔍 Search")

        query_translated = translate_to_english(query)

    if search_button:
        progress_bar = st.progress(0)
        progress_bar.progress(10)
        progress_bar.progress(30)

        recommended_papers = search_papers(query_translated.lower(), combined_data)

        progress_bar.progress(70)
        progress_bar.progress(100)

        def display_cards(recommended_papers):
            for index, i in enumerate(recommended_papers):
                flag, check_url = check_scihub_availability(i["Link"])
                availability_text = ""

                if flag:
                    availability_text = '<p style="color: green; font-weight: semibold;">Available on Sci-Hub</p>'
                else:
                    availability_text = '<p style="color: red; font-weight: semibold;">Not on Sci-Hub</p>'
                with st.container():
                    st.markdown(
                        f"""
                        <div class="card">
                            <a href="{check_url}" target="_blank" class="card-title">{i['Title']}</a>
                            <div class="card-description">
                                {availability_text}
                                <p>Authors : {i['Authors']}</p>
                                <p>Journal : {i['Journal']}</p>
                                <p>Citations : {i['Citations']}</p>
                                <p>{i['Abstract'][:200]}...</p>
                                <details>
                                    <summary>Read More</summary>
                                    <p>{i['Abstract']}</p>
                                </details>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )



        display_cards(recommended_papers)
    else:
        if search_button:
            st.write("No results found for the query. Please try again.")

elif selected == "Analyze":
    st.write("Upload a PDF research paper to analyze its content.")

    # Add radio buttons for selecting the type of user
    user_type = st.radio(
        "Are you a student or a researcher?", ("Student", "Researcher")
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:

        def extract_text_from_pdf(file):
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
            return text

        def summarize_text_with_gemini(api_key, text, user_type):
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel("gemini-pro")

            # Different prompts based on user type
            if user_type == "Student":
                prompt = f"Generate a simple summary from this text in at least 250 words, provide pros and cons in a simple language that is understandable to a student: {text}"
            else:  # Researcher
                prompt = f"Generate a DETAILED summary from this text in at least 500 words, then provide pros and cons, and THEN also give me suggestions for improvement: {text}"

            try:
                response = model.generate_content(prompt)
                return response
            except Exception as e:
                st.error(f"Error during Gemini API call: {e}")
                return None

        def extract_formatted_text(api_response):
            if api_response is None:
                return "Error: No response from API."

            try:
                text_content = api_response.text

                cleaned_text = re.sub(r"\*+", "", text_content)

                return (
                    cleaned_text
                    if cleaned_text
                    else "Error: No text found in response."
                )
            except (KeyError, IndexError, TypeError) as e:
                st.error(f"Error extracting text: {e}")
                return "Error extracting text."

        # Extract text from the uploaded PDF
        st.info("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            st.success("Text extracted successfully.")

            # Summarize the text using Gemini API based on user type
            api_key = "AIzaSyDwOvC6nJpM9XEkc_PbqhbWKq5Tzdx2xFI"
            api_response = summarize_text_with_gemini(api_key, pdf_text, user_type)

            if api_response:
                formatted_summary = extract_formatted_text(api_response)
                st.subheader("Summary:")
                st.write(formatted_summary)

                # Create a downloadable file in memory (using BytesIO)
                summary_file = io.BytesIO()
                summary_file.write(formatted_summary.encode("utf-8"))
                summary_file.seek(0)

                # Provide a download button for the summary
                st.download_button(
                    label="Download Summary",
                    data=summary_file,
                    file_name="research_paper_summary.txt",
                    mime="text/plain",
                )
        else:
            st.error("Failed to extract text from the uploaded PDF.")

# About Page
elif selected == "Compare":
    st.write("### Compare Research Papers")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file containing research papers", type="csv")

    if uploaded_file is not None:
        # Load papers from the uploaded CSV file
        all_papers = pd.read_csv(uploaded_file)  # Read the uploaded file into a DataFrame

        if all_papers.empty:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        else:
            paper_titles = all_papers["Title"].tolist()
            selected_titles = st.multiselect("Select papers to compare:", paper_titles)

            if selected_titles:  # Check if any titles are selected
                if st.button("Compare Selected Papers"):
                    result_df = compare_papers(selected_titles, all_papers)
                    
                    if result_df is not None:
                        st.write("### Paper Comparison")

                        # Create a comparison table
                        comparison_table = pd.DataFrame(columns=["Attribute"] + selected_titles)

                        attributes = ['Title', 'Authors', 'Citations', 'Journal', 'Publication Date']

                        for attr in attributes:
                            row_data = [attr] + [result_df.loc[result_df['Title'] == title, attr].values[0] if not result_df.loc[result_df['Title'] == title, attr].empty else "N/A" for title in selected_titles]
                            comparison_table.loc[len(comparison_table)] = row_data

                        st.table(comparison_table)
            else:
                st.info("Please select at least one paper to compare.")
    else:
        st.info("Please upload a CSV file to get started.")


elif selected == "About":
    st.write("This app helps researchers and students find relevant research papers based on their search queries. Powered by GenAI, it provides results tailored to your interests.")