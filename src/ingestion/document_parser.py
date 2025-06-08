import fitz  # PyMuPDF
import pdfplumber
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import OrderedDict
from ..config import settings
from ..utils.logger_config import setup_logger

# Setup a logger specific to this module
logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')

# Download necessary NLTK data
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class TextProcessor:
    """
    A class to handle text processing tasks including PDF extraction,
    text cleaning, and chunking.
    """

    def __init__(self):
        """
        Initializes the TextProcessor with a RecursiveCharacterTextSplitter.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file using PyMuPDF.
        """
        logger.info(f"Extracting text from {pdf_path} using PyMuPDF...")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def extract_text_from_pdf_alternative(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file using pdfplumber (alternative).
        """
        logger.info(f"Extracting text from {pdf_path} using pdfplumber...")
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def read_text_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Reads text from a plain text file.
        """
        logger.info(f"Reading text from {file_path}...")
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()

    def clean_text(
        self,
        text: str,
        stem_method: str = "spacy",
        custom_stopwords: list = None,
        remove_consecutive_words: bool = True,
        remove_duplicate_sentences: bool = True,
    ) -> str:
        """
        Cleans the input text with various options.
        """
        logger.info(f"Cleaning text with the following options:")

        # --- Sentence-level cleaning (if requested) ---
        if remove_duplicate_sentences:
            sentences = sent_tokenize(text)
            # Use OrderedDict to preserve order while getting unique sentences
            unique_sentences = list(OrderedDict.fromkeys(sentences))
            text = " ".join(unique_sentences)

        # --- Basic text cleaning ---
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\n", " ", text)
        words = text.split()

        # --- Consecutive word removal (if requested) ---
        if remove_consecutive_words:
            if not words:
                return ""
            non_consecutive_words = [words[0]]
            for i in range(1, len(words)):
                if words[i] != words[i-1]:
                    non_consecutive_words.append(words[i])
            words = non_consecutive_words

        # --- Stopword removal ---
        stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        words = [word for word in words if word not in stop_words]

        # --- Lemmatization or Stemming ---
        if stem_method == "spacy":
            doc = nlp(" ".join(words))
            lemmatized_words = [token.lemma_ for token in doc]
            final_text = " ".join(lemmatized_words)
        elif stem_method == "nltk_lem":
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            final_text = " ".join(lemmatized_words)
        elif stem_method == "nltk_stem":
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in words]
            final_text = " ".join(stemmed_words)
        else:
            final_text = " ".join(words)

        return final_text

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunks the text into smaller pieces using LangChain's RecursiveCharacterTextSplitter.
        """
        logger.info("Chunking text...")
        chunks = self.text_splitter.split_text(text)
        return chunks