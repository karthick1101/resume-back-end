from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import matcher

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match")
async def match(resume: UploadFile = File(...), jd: UploadFile = File(...)):
    resume_bytes = await resume.read()
    jd_bytes = await jd.read()

    text_resume = matcher.extract_text_from_pdf(resume_bytes)
    text_jd = matcher.extract_text_from_pdf(jd_bytes)

    clean_resume = matcher.preprocess(text_resume)
    clean_jd = matcher.preprocess(text_jd)

    match_score = float(matcher.get_similarity(clean_resume, clean_jd))

    skills_resume = matcher.extract_skills(clean_resume)
    skills_jd = matcher.extract_skills(clean_jd)

    matched = sorted(skills_resume & skills_jd)
    missing = sorted(skills_jd - skills_resume)
    suggestions = matcher.generate_improvement_tips(missing, clean_resume)

    return {
        "match_score": match_score / 100,  # To return as 0.85 etc.
        "matched_skills": matched,
        "missing_skills": missing,
        "suggestions": suggestions
    }
