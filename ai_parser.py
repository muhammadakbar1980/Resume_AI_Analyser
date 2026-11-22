"""
STEP 3 - AI EXTRACTION  (uses Groq AI)

Reads clean resume text and clean job description text and returns
structured details as Python dictionaries.
This file is fully independent: it does not import any other project file.
The prompts are written inside this file.

Main functions:
    parse_resume(resume_text, api_key, model) -> (data_dict, error)
    parse_job_description(jd_text, api_key, model) -> (data_dict, error)
"""

import json
import os
import re

DEFAULT_MODEL = "openai/gpt-oss-120b"
BACKUP_MODEL = "openai/gpt-oss-20b"
MAX_INPUT_CHARS = 15000
DEGREE_LEVELS = ["none", "high_school", "diploma", "bachelor", "master", "phd"]

# ---------------------------------------------------------------- prompts

RESUME_PROMPT = """You extract information from a RESUME.
Return ONLY a JSON object with exactly these keys:

{
  "name": "candidate name or empty string",
  "skills": ["list of technical and soft skills, tools, software, languages"],
  "total_experience_years": number (total years of work experience; 0 if none; estimate from dates if needed),
  "job_titles": ["job titles held"],
  "education": ["degrees with subject, e.g. 'Bachelor of Science in Computer Science'"],
  "highest_degree_level": one of "none", "high_school", "diploma", "bachelor", "master", "phd",
  "certifications": ["certifications and licences"],
  "projects": ["short one-line description of each project"]
}

Rules:
- Use only information that is written in the resume. Do not invent anything.
- Write each skill as a short standard name, e.g. "Python", "Machine Learning", "Microsoft Excel", "Project Management".
- Expand abbreviations where clear, e.g. "ML" -> "Machine Learning", "JS" -> "JavaScript".
- If something is not found, use an empty list, 0 or empty string.
"""

JD_PROMPT = """You extract requirements from a JOB DESCRIPTION.
Return ONLY a JSON object with exactly these keys:

{
  "job_title": "job title or empty string",
  "must_have_skills": ["skills, tools and knowledge that are required"],
  "nice_to_have_skills": ["skills that are preferred, a plus, or optional"],
  "required_experience_years": number (minimum years required; 0 if not stated),
  "required_degree_level": one of "none", "high_school", "diploma", "bachelor", "master", "phd",
  "required_qualifications": ["degrees or fields of study asked for"],
  "required_certifications": ["certifications asked for"],
  "ats_keywords": ["10 to 25 important keywords and short phrases an ATS would scan for"]
}

Rules:
- Use only information written in the job description. Do not invent anything.
- Write each skill as a short standard name, e.g. "Python", "Machine Learning", "Microsoft Excel".
- Expand abbreviations where clear, e.g. "ML" -> "Machine Learning".
- If the text says "or equivalent experience", still record the degree level.
- If something is not found, use an empty list, 0 or "none".
"""

RESUME_DEFAULTS = {
    "name": "",
    "skills": [],
    "total_experience_years": 0,
    "job_titles": [],
    "education": [],
    "highest_degree_level": "none",
    "certifications": [],
    "projects": [],
}

JD_DEFAULTS = {
    "job_title": "",
    "must_have_skills": [],
    "nice_to_have_skills": [],
    "required_experience_years": 0,
    "required_degree_level": "none",
    "required_qualifications": [],
    "required_certifications": [],
    "ats_keywords": [],
}

# ---------------------------------------------------------------- helpers


def _get_api_key(api_key):
    if api_key:
        return api_key
    return os.environ.get("GROQ_API_KEY", "")


def _extract_json(text):
    """Find and load the JSON object in the AI reply."""
    if not text:
        raise ValueError("Empty reply from AI")
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _to_number(value):
    try:
        if isinstance(value, str):
            found = re.findall(r"\d+(?:\.\d+)?", value)
            return float(found[0]) if found else 0.0
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [part for part in re.split(r"[,;\n]", value)]
    if not isinstance(value, list):
        return []
    result, seen = [], set()
    for item in value:
        item = str(item).strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def _validate(data, defaults):
    """Make sure every expected key exists with the right type."""
    clean = {}
    for key, default in defaults.items():
        value = data.get(key, default) if isinstance(data, dict) else default
        if isinstance(default, list):
            clean[key] = _to_list(value)
        elif isinstance(default, (int, float)):
            clean[key] = _to_number(value)
        elif key.endswith("degree_level"):
            value = str(value).strip().lower().replace(" ", "_")
            clean[key] = value if value in DEGREE_LEVELS else "none"
        else:
            clean[key] = str(value or "").strip()
    return clean


def _ask_ai(system_prompt, user_text, api_key, model):
    """Call Groq. Tries the chosen model, retries once, then the backup model."""
    from groq import Groq

    key = _get_api_key(api_key)
    if not key:
        raise ValueError("Groq API key is missing.")
    client = Groq(api_key=key)

    models_to_try = [model or DEFAULT_MODEL]
    if BACKUP_MODEL not in models_to_try:
        models_to_try.append(BACKUP_MODEL)

    last_error = None
    for current_model in models_to_try:
        for _attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text[:MAX_INPUT_CHARS]},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                return _extract_json(response.choices[0].message.content)
            except Exception as error:
                last_error = error
    raise RuntimeError(f"AI extraction failed: {last_error}")


# ---------------------------------------------------------------- main functions


def parse_resume(resume_text, api_key=None, model=DEFAULT_MODEL):
    """Returns (resume_data, error). error is "" when successful."""
    if not resume_text or not resume_text.strip():
        return dict(RESUME_DEFAULTS), "Resume text is empty."
    try:
        raw = _ask_ai(RESUME_PROMPT, "RESUME:\n" + resume_text, api_key, model)
        return _validate(raw, RESUME_DEFAULTS), ""
    except Exception as error:
        return dict(RESUME_DEFAULTS), str(error)


def parse_job_description(jd_text, api_key=None, model=DEFAULT_MODEL):
    """Returns (jd_data, error). error is "" when successful."""
    if not jd_text or not jd_text.strip():
        return dict(JD_DEFAULTS), "Job description text is empty."
    try:
        raw = _ask_ai(JD_PROMPT, "JOB DESCRIPTION:\n" + jd_text, api_key, model)
        return _validate(raw, JD_DEFAULTS), ""
    except Exception as error:
        return dict(JD_DEFAULTS), str(error)
