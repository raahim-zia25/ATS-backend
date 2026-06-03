from fastapi import FastAPI, Response, UploadFile, File, Form

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from groq import Groq

from fpdf import FPDF

import os

import base64

import json

import io

from pypdf import PdfReader

from typing import List

try:

    from dotenv import load_dotenv

except ImportError:

    load_dotenv = None


if load_dotenv is not None:

    load_dotenv()

else:

    # Fallback parser so local .env still works without python-dotenv installed.

    try:

        with open(".env", "r", encoding="utf-8") as env_file:

            for raw_line in env_file:

                line = raw_line.strip()

                if not line or line.startswith("#") or "=" not in line:

                    continue

                key, value = line.split("=", 1)

                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    except FileNotFoundError:

        pass


app = FastAPI()


# Add fixed local origins and allow custom origins from env for production frontends.

default_origins = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

extra_origins = {
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}

allowed_origins = sorted(default_origins | extra_origins)


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Supports Vercel preview/production frontend URLs.
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY, timeout=60.0) if GROQ_API_KEY else None


def get_groq_client() -> Groq:

    if client is None:

        raise RuntimeError("Missing GROQ_API_KEY. Set it in environment variables.")

    return client


@app.get("/health")
async def health_check():

    return {
        "status": "ok",
        "service": "proposal-agent",
        "groq_configured": bool(GROQ_API_KEY),
    }


class ProposalRequest(BaseModel):

    job_description: str


class CVRequest(BaseModel):

    personal_details: str


class MatchRequest(BaseModel):

    cv_text: str

    job_description: str


# New interactive injection payload model


class InjectionRequest(BaseModel):

    cv_text: str

    skills_to_add: List[str]


class ExactLayoutCV(FPDF):

    def footer(self):

        self.set_y(-15)

        self.set_font("Helvetica", "I", 8)

        self.set_text_color(148, 163, 184)

        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "R")


def clean_unicode_to_ascii(text: str) -> str:

    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }

    for bad_char, good_char in replacements.items():

        text = text.replace(bad_char, good_char)

    return text.encode("latin-1", "ignore").decode("latin-1")


def build_pdf_canvas(sanitized_cv_text: str) -> str:
    pdf = ExactLayoutCV(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Clean the text once and split safely
    # We remove the most problematic markdown characters
    clean_text = (
        sanitized_cv_text.replace("**", "").replace("###", "").replace("===", "")
    )
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]

    # 2. Iterate safely
    for line in lines:
        # A. Identity Headers (Assume top 2-3 lines are title/info)
        if len(line) < 50 and (line.isupper() or "@" in line):
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, line, ln=True, align="L")

        # B. Section Headers (Look for common patterns)
        elif any(
            header in line.upper()
            for header in ["EXPERIENCE", "SKILLS", "EDUCATION", "PROJECTS", "SUMMARY"]
        ):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, line.upper(), ln=True)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2)

        # C. Everything else (Bullets or body text)
        else:
            pdf.set_font("Helvetica", "", 10)
            if line.startswith("-") or line.startswith("•"):
                pdf.multi_cell(0, 5, "  " + line)
            else:
                pdf.multi_cell(0, 5, line)
            pdf.ln(1)

    # 3. Defensive PDF output
    try:
        # Use dest='S' to get the string/bytes of the PDF directly
        pdf_output = pdf.output(dest="S")
        pdf_bytes = (
            pdf_output if isinstance(pdf_output, bytes) else pdf_output.encode("latin1")
        )
        return base64.b64encode(pdf_bytes).decode("utf-8")
    except Exception as e:
        print(f"PDF Render Error: {e}")
        return ""


@app.post("/generate")
async def generate_proposal(request: ProposalRequest):

    user_job_post = request.job_description

    dynamic_prompt = f"Write a high-converting Upwork proposal for:\n{user_job_post}"

    try:

        groq_client = get_groq_client()

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": dynamic_prompt}],
            temperature=0.7,
            max_tokens=800,
        )

        return {"proposal": completion.choices[0].message.content}

    except Exception as e:

        return {"proposal": f"Error: {str(e)}"}


@app.post("/generate-cv")
async def generate_cv(request: CVRequest):

    user_info = request.personal_details

    cv_blueprint_prompt = f"Format this raw data into an elite technical resume blueprint matching our structural headers exactly.\nRaw Data:\n{user_info}"

    try:

        groq_client = get_groq_client()

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": cv_blueprint_prompt}],
            temperature=0.3,
            max_tokens=1200,
        )

        raw_cv_text = completion.choices[0].message.content

        pdf_base64 = build_pdf_canvas(clean_unicode_to_ascii(raw_cv_text))

        return {"text_preview": raw_cv_text, "pdf_data": pdf_base64}

    except Exception as e:

        return {"error": str(e)}


# CORE FEATURE: INJECTS KEYWORDS STRAIGHT INTO THE ORIGINAL UPLOADED TEXT STREAM


@app.post("/inject-skills-to-cv")
async def inject_skills_to_cv(request: InjectionRequest):
    try:
        # 1. Clean the input text and split into lines
        lines = request.cv_text.split("\n")

        # 2. Add skills safely
        skills_string = ", ".join(request.skills_to_add)
        skills_inserted = False

        # Try to find the section and append
        for i, line in enumerate(lines):
            if "TECHNICAL SKILLS" in line.upper() or "FRAMEWORKS" in line.upper():
                lines.insert(i + 1, f"Frameworks & Libraries: {skills_string}")
                skills_inserted = True
                break

        # If section not found, just append to end
        if not skills_inserted:
            lines.append(f"\nTECHNICAL SKILLS\nFrameworks & Libraries: {skills_string}")

        final_cv_text = "\n".join(lines)

        # 3. CRITICAL: Pass cleaned text to the PDF builder
        # Ensure we are not sending empty or None to the builder
        pdf_base64 = build_pdf_canvas(clean_unicode_to_ascii(final_cv_text))

        if not pdf_base64:
            raise ValueError("PDF builder returned empty stream.")

        return {"pdf_data": pdf_base64, "updated_text": final_cv_text}

    except Exception as e:
        print(f"CRITICAL ERROR in inject-skills-to-cv: {str(e)}")
        return {"error": f"Failed compiling modified variant: {str(e)}"}


@app.post("/match-score")
async def match_score(request: MatchRequest):

    return await run_ats_analysis(request.cv_text, request.job_description)


@app.post("/match-upload")
async def match_upload(job_description: str = Form(...), file: UploadFile = File(...)):

    try:

        file_bytes = await file.read()

        pdf_stream = io.BytesIO(file_bytes)

        reader = PdfReader(pdf_stream)

        extracted_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                extracted_text += text + "\n"

        if not extracted_text.strip():

            return {"error": "Could not extract clear layers from this document."}

        # Parse text structure safely into blueprint strings mapping your resume headers

        blueprint_normalized = f"NAME: RAAHIM ZIA\nSUBTITLE: Frontend Developer\nCONTACT: Karachi, Pakistan\n\nTECHNICAL SKILLS\n{extracted_text}"

        # Analyze using standard flow

        ats_analysis = await run_ats_analysis(extracted_text, job_description)

        # Pass along raw text layers so the client keeps track of original layout values

        ats_analysis["extracted_cv_text"] = blueprint_normalized

        return ats_analysis

    except Exception as e:

        return {"error": f"Failed analyzing vector formats: {str(e)}"}


async def run_ats_analysis(cv_text: str, job_description: str):
    ats_prompt = f"""
    You are an advanced Applicant Tracking System. Evaluate this candidate's CV against the targeting requirements inside the Job Description.

    Candidate CV: {cv_text}
    Target Job Description: {job_description}

    Output valid JSON only. NO markdown, NO backticks. 
    Expected structure:
    {{
        "score": 85,
        "strong_matches": ["React.js", "Next.js"],
        "missing_skills": ["Docker", "AWS"],
        "suggestions": ["Add missing technologies to your profile"]
    }}
    """
    try:
        groq_client = get_groq_client()
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": ats_prompt}],
            temperature=0.2,
            max_tokens=800,
        )

        content = completion.choices[0].message.content.strip()

        # --- THE FIX: Clean the response ---
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(content)

    except Exception as e:
        print(f"DEBUG: JSON Parse Error: {e}")  # Check terminal for the raw error
        return {
            "score": 0,
            "strong_matches": [],
            "missing_skills": [],
            "suggestions": ["A"
            ""
            ""
            "I failed to return valid JSON format."],
        }
