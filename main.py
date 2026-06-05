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
        "•": "-",
    }
    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)
    return text.encode("latin-1", "ignore").decode("latin-1")


def build_pdf_canvas(sanitized_cv_text: str) -> str:
    pdf = ExactLayoutCV(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    clean_text = (
        sanitized_cv_text.replace("**", "")
        .replace("###", "")
        .replace("===", "")
        .strip()
    )
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]

    standard_headers = [
        "ABOUT ME",
        "PROFESSIONAL SUMMARY",
        "WORK EXPERIENCE",
        "EXPERIENCE",
        "SKILLS",
        "TECHNICAL SKILLS",
        "EDUCATION",
        "PROJECTS",
        "KEY HIGHLIGHTS",
        "CERTIFICATIONS",
        "LANGUAGES",
        "INTERESTS",
        "SUMMARY",
    ]

    for i, line in enumerate(lines):
        upper_line = line.upper().strip()

        # 1. Identity Headers (Fixed Top 3 Lines)
        if i == 0:
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 10, line.upper(), ln=True, align="C")
        elif i == 1:
            pdf.set_font("Helvetica", "I", 14)
            pdf.set_text_color(79, 70, 229)
            pdf.cell(0, 8, line, ln=True, align="C")
        elif i == 2:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 6, line, ln=True, align="C")
            pdf.ln(6)
            pdf.set_text_color(0, 0, 0)

        # 2. Main Section Headers
        elif upper_line in standard_headers:
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 8, upper_line, ln=True)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
            pdf.ln(3)
            pdf.set_text_color(0, 0, 0)

        # 3. Body Text, Bullets, and Sub-skills
        else:
            pdf.set_font("Helvetica", "", 10)
            if line.startswith("-") or line.startswith("*"):
                pdf.set_x(20)
                pdf.multi_cell(0, 5, line)
                pdf.set_x(15)
            elif ":" in line and len(line.split(":")[0]) < 30:
                pdf.set_font("Helvetica", "B", 10)
                parts = line.split(":", 1)
                pdf.write(5, parts[0] + ": ")
                pdf.set_font("Helvetica", "", 10)
                pdf.write(5, parts[1].strip() + "\n")
            else:
                pdf.multi_cell(0, 5, line)
            pdf.ln(1.5)

    try:
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

    # 1. Dynamically Load Real-Time Data (Portfolio & Examples)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    portfolio_path = os.path.join(base_dir, "src", "data", "portfolio.json")
    examples_path = os.path.join(base_dir, "src", "data", "examples.json")

    portfolio_context = "No specific portfolio data provided. Focus on writing a strong, general professional proposal."
    examples_context = "Tone should be professional, concise, and confident."

    try:
        if os.path.exists(portfolio_path):
            with open(portfolio_path, "r", encoding="utf-8") as f:
                portfolio_context = f.read()
        if os.path.exists(examples_path):
            with open(examples_path, "r", encoding="utf-8") as f:
                examples_context = f.read()
    except Exception as e:
        print(f"DEBUG: Error loading data files: {e}")

    # 2. Smarter System Prompt (Anti-Code, but allows general proposals)
    system_prompt = """You are an expert Upwork proposal writer. 
    CRITICAL RULE 1: Analyze the user input. If the input is purely programming code (Python, React, etc.), a stack trace, or random keyboard gibberish, you MUST abort and output EXACTLY and ONLY the string 'INVALID_INPUT_ERROR'.
    CRITICAL RULE 2: If it IS a valid job description, write a customized proposal. Use the 'Portfolio Data' below to highlight relevant experience. If no specific portfolio data is provided, write a strong general proposal. Do not invent fake names."""

    dynamic_prompt = f"""
    --- My Real Portfolio Data ---
    {portfolio_context}

    --- My Writing Style Examples ---
    {examples_context}

    --- Target Client Job Description ---
    {user_job_post}
    """

    try:
        groq_client = get_groq_client()
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dynamic_prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        response_text = completion.choices[0].message.content.strip()

        # Intercept the invalid error flag
        if "INVALID_INPUT_ERROR" in response_text:
            return {
                "error": "Invalid input detected. Please paste an actual client job description, not code or gibberish."
            }

        return {"proposal": response_text}
    except Exception as e:
        return {"error": f"Error generating proposal: {str(e)}"}


@app.post("/generate-cv")
async def generate_cv(request: CVRequest):
    user_info = request.personal_details

    system_prompt = """You are a strict data validation and formatting engine. 
    CRITICAL RULE 1: Analyze the user's input. If the data consists of programming code (like React, HTML, Python), random gibberish, or lacks real personal/professional context (like a real name or work history), you MUST reply with EXACTLY the word "INVALID_DATA" and nothing else.
    CRITICAL RULE 2: Do NOT invent, hallucinate, or use fake placeholder names. Do not output conversational text."""

    cv_blueprint_prompt = f"""If the data is valid, format it into an elite technical resume blueprint matching this EXACT structure. Do not use markdown bolding (**).

[YOUR NAME]
[YOUR SUBTITLE]
[YOUR CONTACT INFO: Phone | Email | Location]

ABOUT ME
[1 paragraph summary]

WORK EXPERIENCE
[Job Title] | [Company] | [Dates]
* [Achievement 1]
* [Achievement 2]

PROJECTS
[Project Name]
* [Details]

TECHNICAL SKILLS
Core Languages: [Languages]
Frameworks & Libraries: [Frameworks]
Tools & Workflow: [Tools]

EDUCATION
[Degree]
[Institution] | [Dates]

KEY HIGHLIGHTS
* [Highlight 1]

Raw Data:
{user_info}"""
    try:
        groq_client = get_groq_client()
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cv_blueprint_prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        raw_cv_text = completion.choices[0].message.content.strip()

        if "INVALID_DATA" in raw_cv_text:
            return {
                "error": "Invalid input detected. Please provide real resume details, not code or random text."
            }

        pdf_base64 = build_pdf_canvas(clean_unicode_to_ascii(raw_cv_text))
        return {"text_preview": raw_cv_text, "pdf_data": pdf_base64}
    except Exception as e:
        return {"error": str(e)}


@app.post("/inject-skills-to-cv")
async def inject_skills_to_cv(request: InjectionRequest):
    try:
        lines = request.cv_text.split("\n")
        skills_string = ", ".join(request.skills_to_add)
        skills_inserted = False

        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if (
                "FRAMEWORKS & LIBRARIES" in line_upper
                or "FRAMEWORKS" in line_upper
                or "CORE LANGUAGES" in line_upper
            ) and ":" in line_upper:
                label_part = line_upper.split(":")[0]
                if len(label_part) < 35:
                    lines[i] = f"{line}, {skills_string}"
                    skills_inserted = True
                    break

        if not skills_inserted:
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                if line_upper == "TECHNICAL SKILLS" or line_upper == "SKILLS":
                    lines.insert(i + 1, f"Frameworks & Libraries: {skills_string}")
                    skills_inserted = True
                    break

        if not skills_inserted:
            lines.append("")
            lines.append(f"TECHNICAL SKILLS")
            lines.append(f"Frameworks & Libraries: {skills_string}")

        final_cv_text = "\n".join(lines)
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
            return {
                "error": "Could not extract clear layers from this document. Make sure it is a valid text-based PDF."
            }

        clean_content = extracted_text.strip()
        ats_analysis = await run_ats_analysis(clean_content, job_description)

        if "error" in ats_analysis:
            return ats_analysis

        ats_analysis["extracted_cv_text"] = clean_content

        return ats_analysis
    except Exception as e:
        return {"error": f"Failed analyzing vector formats: {str(e)}"}


async def run_ats_analysis(cv_text: str, job_description: str):
    system_prompt = """You are a ruthless, highly critical Applicant Tracking System (ATS).
    
    CRITICAL RULE 1: Read the Candidate CV Context first. If the CV contains random code, gibberish, an error message/stack trace, or completely lacks standard resume elements (like work history or technical skills), you MUST output EXACTLY and ONLY the string 'INVALID_DATA'.
    CRITICAL RULE 2: NEVER invent matches. A skill is a "strong_match" ONLY if it appears in BOTH the CV and the Job Description. 
    CRITICAL RULE 3: If a skill is in the Job Description but missing from the CV, it MUST go into "missing_skills". Never assume the candidate has it.
    CRITICAL RULE 4: ONLY output raw JSON. No formatting, no backticks, no prose."""

    ats_prompt = f"""
    --- CANDIDATE CV START ---
    {cv_text}
    --- CANDIDATE CV END ---

    --- TARGET JOB DESCRIPTION START ---
    {job_description}
    --- TARGET JOB DESCRIPTION END ---

    Expected structure (leave score as 0, it will be calculated automatically):
    {{
        "score": 0,
        "strong_matches": [],
        "missing_skills": [],
        "suggestions": []
    }}
    """
    try:
        groq_client = get_groq_client()
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ats_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        content = completion.choices[0].message.content.strip()

        if "INVALID_DATA" in content:
            return {
                "error": "Invalid resume detected. The uploaded document does not appear to be a legitimate, complete CV."
            }

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)

        # --- PYTHON MATHEMATICAL SCORE CALCULATION ---
        matched_count = len(result.get("strong_matches", []))
        missing_count = len(result.get("missing_skills", []))
        total_skills = matched_count + missing_count

        if total_skills > 0:
            calculated_score = int((matched_count / total_skills) * 100)
            result["score"] = calculated_score
        else:
            result["score"] = 0

        return result
    except Exception as e:
        print(f"DEBUG: JSON Parse Error: {e}")
        return {"error": f"I failed to return valid JSON format. Error: {str(e)}"}
