"""
STEP 4 - MATCHING, ATS CHECK & SCORING  (pure Python, no AI)

Compares resume data with job description data and calculates scores.
The same input always gives the same score.
This file is fully independent: it does not import any other project file.

Main function:
    analyse(resume_data, jd_data, resume_text) -> results_dict
"""

import re

WEIGHTS = {"skills": 40, "experience": 25, "keywords": 20, "qualification": 15}

DEGREE_RANK = {"none": 0, "high_school": 1, "diploma": 2,
               "bachelor": 3, "master": 4, "phd": 5}

# Common short forms and different spellings -> one standard form
SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "advanced excel": "excel",
    "ms word": "word",
    "microsoft word": "word",
    "ms office": "microsoft office",
    "powerbi": "power bi",
    "microsoft power bi": "power bi",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "node": "node.js",
    "nodejs": "node.js",
    "react.js": "react",
    "reactjs": "react",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "pm": "project management",
    "communication skills": "communication",
    "team work": "teamwork",
    "team player": "teamwork",
    "problem-solving": "problem solving",
}


# ---------------------------------------------------------------- helpers


def normalise(term):
    """Lower case, remove extra symbols, map synonyms."""
    term = str(term or "").lower().strip()
    term = re.sub(r"\(.*?\)", "", term)               # remove brackets text
    term = re.sub(r"[^a-z0-9+#./\s-]", " ", term)     # keep c++, c#, node.js
    term = re.sub(r"\s+", " ", term).strip(" .-/")
    return SYNONYMS.get(term, term)


def _in_text(term, text):
    """True if the term appears in the text as a whole word/phrase."""
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _is_matched(required, resume_terms, resume_text_norm):
    req = normalise(required)
    if not req:
        return False
    for have in resume_terms:
        if req == have:
            return True
        # "python" matches "python programming" (only for words of 3+ letters)
        if len(req) >= 3 and len(have) >= 3 and (_in_text(req, have) or _in_text(have, req)):
            return True
    return _in_text(req, resume_text_norm)


def _match_list(required_list, resume_terms, resume_text_norm):
    matched, missing = [], []
    for item in required_list or []:
        if _is_matched(item, resume_terms, resume_text_norm):
            matched.append(item)
        else:
            missing.append(item)
    return matched, missing


def _normalise_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Also map synonyms inside the text, so "ML" in the resume counts
    for short, full in SYNONYMS.items():
        text = re.sub(r"(?<![a-z0-9])" + re.escape(short) + r"(?![a-z0-9])", f"{short} {full}", text)
    return text


def _ats_format_checks(resume_text):
    """Simple checks that help a resume pass ATS software."""
    issues = []
    text = resume_text or ""
    lower = text.lower()
    words = len(text.split())

    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        issues.append("No email address found.")
    if not re.search(r"(\+?\d[\d\s().-]{8,}\d)", text):
        issues.append("No phone number found.")

    headings = {
        "Experience": ["experience", "employment", "work history"],
        "Education": ["education", "qualification", "academic"],
        "Skills": ["skills", "competencies", "expertise"],
    }
    for heading, options in headings.items():
        if not any(option in lower for option in options):
            issues.append(f"No clear '{heading}' section heading found.")

    if words < 200:
        issues.append(f"Resume is short ({words} words). Most good resumes have 300-900 words.")
    elif words > 1500:
        issues.append(f"Resume is long ({words} words). Try to keep it to 1-2 pages.")

    if not re.search(r"\d+\s?%|\$\s?\d|\b\d{2,}\b", text):
        issues.append("No numbers or measurable results found (e.g. 'increased sales by 20%').")
    return issues


def _verdict(score):
    if score >= 75:
        return "Strong match"
    if score >= 50:
        return "Moderate match"
    return "Weak match"


# ---------------------------------------------------------------- main function


def analyse(resume_data, jd_data, resume_text=""):
    """Compare resume with job description. Returns a results dictionary."""
    resume_data = resume_data or {}
    jd_data = jd_data or {}
    resume_text_norm = _normalise_text(resume_text)

    resume_terms = {normalise(s) for s in
                    (resume_data.get("skills", []) + resume_data.get("certifications", []))}
    resume_terms.discard("")

    # ---- Skills (must-have counts double)
    must = jd_data.get("must_have_skills", [])
    nice = jd_data.get("nice_to_have_skills", [])
    must_matched, must_missing = _match_list(must, resume_terms, resume_text_norm)
    nice_matched, nice_missing = _match_list(nice, resume_terms, resume_text_norm)
    total_points = 2 * len(must) + len(nice)
    if total_points == 0:
        skills_score = 100.0
    else:
        skills_score = 100.0 * (2 * len(must_matched) + len(nice_matched)) / total_points

    # ---- Experience
    have_years = float(resume_data.get("total_experience_years", 0) or 0)
    need_years = float(jd_data.get("required_experience_years", 0) or 0)
    if need_years <= 0:
        experience_score = 100.0
    else:
        experience_score = min(100.0, 100.0 * have_years / need_years)
    experience_gap = max(0.0, need_years - have_years)

    # ---- Keywords
    keywords = jd_data.get("ats_keywords", [])
    kw_found, kw_missing = _match_list(keywords, resume_terms, resume_text_norm)
    keyword_score = 100.0 if not keywords else 100.0 * len(kw_found) / len(keywords)

    # ---- Qualification (degree level + certifications)
    have_level = resume_data.get("highest_degree_level", "none")
    need_level = jd_data.get("required_degree_level", "none")
    have_rank = DEGREE_RANK.get(have_level, 0)
    need_rank = DEGREE_RANK.get(need_level, 0)
    degree_ok = have_rank >= need_rank
    if need_rank == 0:
        degree_score = 100.0
    elif degree_ok:
        degree_score = 100.0
    else:
        degree_score = 100.0 * have_rank / need_rank * 0.5   # partial credit

    certs_needed = jd_data.get("required_certifications", [])
    cert_matched, cert_missing = _match_list(certs_needed, resume_terms, resume_text_norm)
    if certs_needed:
        cert_score = 100.0 * len(cert_matched) / len(certs_needed)
        qualification_score = 0.7 * degree_score + 0.3 * cert_score
    else:
        qualification_score = degree_score

    # ---- Final score
    section_scores = {
        "skills": round(skills_score, 1),
        "experience": round(experience_score, 1),
        "keywords": round(keyword_score, 1),
        "qualification": round(qualification_score, 1),
    }
    final_score = sum(section_scores[k] * WEIGHTS[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    return {
        "final_score": final_score,
        "verdict": _verdict(final_score),
        "section_scores": section_scores,
        "weights": dict(WEIGHTS),
        "matched_must_have": must_matched,
        "missing_must_have": must_missing,
        "matched_nice_to_have": nice_matched,
        "missing_nice_to_have": nice_missing,
        "keywords_found": kw_found,
        "keywords_missing": kw_missing,
        "experience": {
            "candidate_years": have_years,
            "required_years": need_years,
            "gap_years": experience_gap,
        },
        "qualification": {
            "candidate_degree": have_level,
            "required_degree": need_level,
            "degree_ok": degree_ok,
            "certifications_matched": cert_matched,
            "certifications_missing": cert_missing,
        },
        "ats_format_issues": _ats_format_checks(resume_text),
        "job_title": jd_data.get("job_title", ""),
        "candidate_name": resume_data.get("name", ""),
    }
