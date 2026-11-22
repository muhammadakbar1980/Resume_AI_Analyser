"""
STEP 5 - AI RECOMMENDATIONS  (uses Groq AI, with a Python backup)

Turns the scoring results into clear problems, practical recommendations
and a short summary. If the AI fails, simple rule-based advice is returned
so the app still works.
This file is fully independent: it does not import any other project file.

Main function:
    get_advice(results, resume_text, api_key, model) -> (advice_dict, error)
"""

import json
import os
import re

DEFAULT_MODEL = "openai/gpt-oss-120b"
BACKUP_MODEL = "openai/gpt-oss-20b"
MAX_RESUME_CHARS = 8000

ADVICE_PROMPT = """You are a professional career coach and ATS (Applicant Tracking System) expert.
You receive (1) the analysis results of a resume compared with a job description and
(2) the resume text.

Return ONLY a JSON object with exactly these keys:
{
  "problems": ["5 to 8 clear problems in simple English, most serious first"],
  "recommendations": ["6 to 10 specific, practical actions the candidate can take"],
  "summary": "2 to 3 sentences explaining the overall result"
}

Rules:
- Base everything on the analysis results. Do not invent missing skills.
- Never advise the candidate to claim skills or experience they do not have.
  For a missing skill, advise to add it ONLY if they truly have it, otherwise to learn it
  (suggest a short course or project).
- Make recommendations specific, e.g. "Add 'Power BI' to your Skills section and mention
  the dashboard project where you used it", not "improve your skills".
- Mention exact missing ATS keywords to include.
- Use the verdict given in the results. Do not change the score.
- Simple English. No markdown symbols inside the strings.
"""


def _get_api_key(api_key):
    if api_key:
        return api_key
    return os.environ.get("GROQ_API_KEY", "")


def _extract_json(text):
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


def _clean_list(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip(" -*•") for item in value if str(item).strip()]


def rule_based_advice(results):
    """Backup advice made by Python rules (used when AI is not available)."""
    results = results or {}
    problems, recs = [], []

    missing_must = results.get("missing_must_have", [])
    missing_nice = results.get("missing_nice_to_have", [])
    kw_missing = results.get("keywords_missing", [])
    exp = results.get("experience", {})
    qual = results.get("qualification", {})

    if missing_must:
        problems.append("Missing required skills: " + ", ".join(missing_must[:10]) + ".")
        recs.append("If you have these skills, add them to your Skills section and show where you used them: "
                    + ", ".join(missing_must[:10]) + ". If not, consider a short course or project to learn them.")
    if missing_nice:
        problems.append("Missing preferred skills: " + ", ".join(missing_nice[:8]) + ".")
        recs.append("Adding preferred skills you genuinely have will make your resume stronger: "
                    + ", ".join(missing_nice[:8]) + ".")
    if kw_missing:
        problems.append("ATS keywords not found in your resume: " + ", ".join(kw_missing[:12]) + ".")
        recs.append("Use the exact wording from the job description where it is true for you, e.g. "
                    + ", ".join(kw_missing[:8]) + ".")
    if exp.get("gap_years", 0) > 0:
        problems.append(f"Experience is below the requirement ({exp.get('candidate_years', 0):g} of "
                        f"{exp.get('required_years', 0):g} years).")
        recs.append("Highlight internships, freelance work, and projects that show relevant experience.")
    if not qual.get("degree_ok", True):
        problems.append(f"Degree level ({qual.get('candidate_degree')}) is below the requirement "
                        f"({qual.get('required_degree')}).")
        recs.append("Mention equivalent experience or certifications that can balance the degree requirement.")
    if qual.get("certifications_missing"):
        problems.append("Missing certifications: " + ", ".join(qual["certifications_missing"]) + ".")
        recs.append("Consider earning these certifications: " + ", ".join(qual["certifications_missing"]) + ".")
    for issue in results.get("ats_format_issues", []):
        problems.append(issue)
    if results.get("ats_format_issues"):
        recs.append("Use a simple one-column layout with clear headings: Summary, Skills, Experience, "
                    "Education, Certifications. Avoid tables, images and text boxes.")
    recs.append("Add measurable results to your work points, e.g. 'reduced processing time by 30%'.")

    if not problems:
        problems.append("No major problems found.")

    score = results.get("final_score", 0)
    verdict = results.get("verdict", "")
    summary = (f"Your resume scored {score}% for this job ({verdict}). "
               "Focus first on the missing required skills and ATS keywords, then on formatting.")
    return {"problems": problems, "recommendations": recs, "summary": summary, "source": "rules"}


def get_advice(results, resume_text="", api_key=None, model=DEFAULT_MODEL):
    """Returns (advice, error). If AI fails, advice comes from rule_based_advice()."""
    key = _get_api_key(api_key)
    if not key:
        return rule_based_advice(results), "Groq API key is missing. Showing basic advice."

    try:
        from groq import Groq
        client = Groq(api_key=key)
    except Exception as error:
        return rule_based_advice(results), f"Could not start Groq client: {error}"

    user_message = (
        "ANALYSIS RESULTS:\n" + json.dumps(results, indent=2, default=str)
        + "\n\nRESUME TEXT:\n" + (resume_text or "")[:MAX_RESUME_CHARS]
    )

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
                        {"role": "system", "content": ADVICE_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                data = _extract_json(response.choices[0].message.content)
                advice = {
                    "problems": _clean_list(data.get("problems")),
                    "recommendations": _clean_list(data.get("recommendations")),
                    "summary": str(data.get("summary", "")).strip(),
                    "source": "ai",
                }
                if advice["problems"] or advice["recommendations"]:
                    if not advice["summary"]:
                        advice["summary"] = rule_based_advice(results)["summary"]
                    return advice, ""
                last_error = "AI returned empty advice"
            except Exception as error:
                last_error = error

    return rule_based_advice(results), f"AI advice failed ({last_error}). Showing basic advice."
