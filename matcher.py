import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define common skills you want to match against
COMMON_SKILLS = {
    'python', 'sql', 'aws', 'nlp', 'machine learning',
    'pandas', 'docker', 'kubernetes', 'tensorflow',
    'data analysis', 'deep learning', 'scikit-learn',
    'numpy', 'matplotlib', 'fastapi', 'django'
}

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using PyMuPDF"""
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess(text: str) -> str:
    """Converts text to lowercase, removes non-alphanumeric characters"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_similarity(text1: str, text2: str) -> float:
    """Computes cosine similarity between two texts"""
    vec1 = model.encode([text1])[0]
    vec2 = model.encode([text2])[0]
    score = cosine_similarity([vec1], [vec2])[0][0]
    return round(score * 100, 2)

def extract_skills(text: str) -> set:
    """Extracts matched skills from text"""
    return {skill for skill in COMMON_SKILLS if skill in text}

def generate_improvement_tips(missing_skills: set, resume_text: str) -> list:
    """Returns suggestions based on missing skills"""
    tips = []
    for skill in missing_skills:
        tips.append(f"Consider adding experience or projects involving '{skill}'.")
    return tips
