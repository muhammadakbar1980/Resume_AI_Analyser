"""
RESUME ANALYSER - main Streamlit app (Step 1: input, Step 6: display)

Run locally:   streamlit run app.py
Each module is loaded separately. If one module has an error, the app
still opens and tells you exactly which file has the problem.
"""

import os
import streamlit as st

st.set_page_config(page_title="Resume Analyser", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- load modules safely
MODULE_ERRORS = {}

try:
    from modules import extractor
except Exception as e:
    extractor = None
    MODULE_ERRORS["modules/extractor.py"] = e

try:
    from modules import ai_parser
except Exception as e:
    ai_parser = None
    MODULE_ERRORS["modules/ai_parser.py"] = e

try:
    from modules import scorer
except Exception as e:
    scorer = None
    MODULE_ERRORS["modules/scorer.py"] = e

try:
    from modules import ai_advisor
except Exception as e:
    ai_advisor = None
    MODULE_ERRORS["modules/ai_advisor.py"] = e

try:
    from modules import report
except Exception as e:
    report = None
    MODULE_ERRORS["modules/report.py"] = e

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


def show_list(items, icon, empty_text):
    if items:
        for item in items:
            st.markdown(f"{icon} {item}")
    else:
        st.caption(empty_text)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key()
    if api_key:
        st.success("Groq API key loaded")
    else:
        api_key = st.text_input("Groq API key", type="password",
                                help="Get a free key at console.groq.com")
    model = st.selectbox("AI model (free on Groq)", MODELS, index=0)
    st.divider()
    st.caption("🔒 Your files are processed in memory and not stored.")
    if MODULE_ERRORS:
        st.error("Some files have errors:")
        for file_name, err in MODULE_ERRORS.items():
            st.code(f"{file_name}: {err}")

# ---------------------------------------------------------------- STEP 1: input
st.title("📄 Resume Analyser")
st.write("Upload your resume and a job description to see how well they match.")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("1. Your resume")
    resume_file = st.file_uploader("Upload resume (PDF, DOCX or TXT)",
                                   type=["pdf", "docx", "txt"], key="resume")
with col_right:
    st.subheader("2. Job description")
    jd_mode = st.radio("How do you want to add it?", ["Paste text", "Upload file"], horizontal=True)
    jd_file, jd_pasted = None, ""
    if jd_mode == "Paste text":
        jd_pasted = st.text_area("Paste job description", height=220,
                                 placeholder="Paste the full job description: skills, experience, qualification...")
    else:
        jd_file = st.file_uploader("Upload job description (PDF, DOCX or TXT)",
                                   type=["pdf", "docx", "txt"], key="jd")

analyse_clicked = st.button("🔍 Analyse", type="primary", use_container_width=True)

# ---------------------------------------------------------------- run pipeline
if analyse_clicked:
    st.session_state.pop("output", None)

    # Input checks
    if extractor is None or ai_parser is None or scorer is None:
        st.error("A required module failed to load. See the sidebar for details.")
        st.stop()
    if not api_key:
        st.error("Please add your Groq API key in the sidebar.")
        st.stop()
    if resume_file is None:
        st.error("Please upload your resume.")
        st.stop()
    if jd_mode == "Paste text" and not jd_pasted.strip():
        st.error("Please paste the job description.")
        st.stop()
    if jd_mode == "Upload file" and jd_file is None:
        st.error("Please upload the job description file.")
        st.stop()

    with st.status("Analysing...", expanded=True) as status:
        # STEP 2: text extraction
        st.write("Reading files...")
        try:
            resume_text, warn = extractor.extract_text(resume_file.getvalue(), resume_file.name)
            if warn:
                st.warning("Resume: " + warn)
            if jd_mode == "Paste text":
                jd_text, jd_warn = extractor.check_pasted_text(jd_pasted, "Job description")
            else:
                jd_text, jd_warn = extractor.extract_text(jd_file.getvalue(), jd_file.name)
            if jd_warn:
                st.warning("Job description: " + jd_warn)
        except Exception as e:
            status.update(label="Failed while reading files", state="error")
            st.error(f"Error in extractor.py: {e}")
            st.stop()
        if not resume_text or not jd_text:
            status.update(label="Could not read text", state="error")
            st.stop()

        # STEP 3: AI extraction
        st.write("AI is reading the resume and job description...")
        try:
            resume_data, err1 = ai_parser.parse_resume(resume_text, api_key, model)
            jd_data, err2 = ai_parser.parse_job_description(jd_text, api_key, model)
        except Exception as e:
            err1, err2 = str(e), ""
        if err1 or err2:
            status.update(label="AI extraction failed", state="error")
            st.error(f"Error in ai_parser.py: {err1 or err2}")
            st.stop()

        # STEP 4: matching & scoring
        st.write("Calculating scores...")
        try:
            results = scorer.analyse(resume_data, jd_data, resume_text)
        except Exception as e:
            status.update(label="Scoring failed", state="error")
            st.error(f"Error in scorer.py: {e}")
            st.stop()

        # STEP 5: AI recommendations (app continues even if this fails)
        st.write("Writing recommendations...")
        advice, advice_error = {"problems": [], "recommendations": [], "summary": ""}, ""
        if ai_advisor is not None:
            try:
                advice, advice_error = ai_advisor.get_advice(results, resume_text, api_key, model)
            except Exception as e:
                advice_error = f"Error in ai_advisor.py: {e}"
        else:
            advice_error = "ai_advisor.py failed to load, so no recommendations."

        status.update(label="Analysis complete ✅", state="complete", expanded=False)

    st.session_state["output"] = {
        "results": results, "advice": advice, "advice_error": advice_error,
        "resume_data": resume_data, "jd_data": jd_data,
    }

# ---------------------------------------------------------------- STEP 6: display
output = st.session_state.get("output")
if output:
    results, advice = output["results"], output["advice"]
    if output["advice_error"]:
        st.info(output["advice_error"])

    st.divider()
    score = results["final_score"]
    verdict = results["verdict"]
    icon = {"Strong match": "✅", "Moderate match": "⚠️", "Weak match": "❌"}.get(verdict, "")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Match score", f"{score}%")
        st.progress(int(score) / 100)
        st.markdown(f"### {icon} {verdict}")
        if results.get("job_title"):
            st.caption(f"Job: {results['job_title']}")
    with c2:
        st.subheader("Score breakdown")
        s = results["section_scores"]
        w = results["weights"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Skills ({w['skills']}%)", f"{s['skills']}%")
        m2.metric(f"Experience ({w['experience']}%)", f"{s['experience']}%")
        m3.metric(f"Keywords ({w['keywords']}%)", f"{s['keywords']}%")
        m4.metric(f"Qualification ({w['qualification']}%)", f"{s['qualification']}%")
        if advice.get("summary"):
            st.info(advice["summary"])

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Matched skills")
        show_list(results["matched_must_have"] + results["matched_nice_to_have"], "🟢", "No matched skills")
    with col_b:
        st.subheader("❌ Missing skills")
        st.markdown("**Required**")
        show_list(results["missing_must_have"], "🔴", "No required skills missing 🎉")
        st.markdown("**Preferred**")
        show_list(results["missing_nice_to_have"], "🟠", "No preferred skills missing")

    st.divider()
    st.subheader("🔑 ATS keywords")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Found in your resume**")
        st.write(", ".join(results["keywords_found"]) or "None")
    with k2:
        st.markdown("**Missing from your resume**")
        st.write(", ".join(results["keywords_missing"]) or "None 🎉")

    exp, qual = results["experience"], results["qualification"]
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"**Experience:** {exp['candidate_years']:g} years "
                    f"(required {exp['required_years']:g})")
    with e2:
        st.markdown(f"**Degree:** {qual['candidate_degree']} (required {qual['required_degree']}) "
                    f"{'✅' if qual['degree_ok'] else '❌'}")

    with st.expander("⚠️ Problems found", expanded=True):
        show_list(advice.get("problems", []), "•", "No problems listed")
    with st.expander("💡 Recommendations", expanded=True):
        for i, rec in enumerate(advice.get("recommendations", []), 1):
            st.markdown(f"**{i}.** {rec}")
    with st.expander("📋 ATS format check"):
        show_list(results["ats_format_issues"], "⚠️", "No format issues found ✅")
    with st.expander("🔎 Details extracted by AI"):
        d1, d2 = st.columns(2)
        d1.markdown("**Resume**")
        d1.json(output["resume_data"])
        d2.markdown("**Job description**")
        d2.json(output["jd_data"])

    if report is not None:
        try:
            report_text = report.build_report(results, advice)
            st.download_button("📥 Download report (TXT)", report_text,
                               file_name="resume_analysis_report.txt", mime="text/plain",
                               use_container_width=True)
        except Exception as e:
            st.warning(f"Report could not be created (report.py): {e}")
    else:
        st.warning("report.py failed to load, so the download is not available.")

    st.caption("Scores are a guide to improve your resume, not a hiring decision.")
