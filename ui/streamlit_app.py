"""Streamlit front-end for the Resume Parser."""
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="wide")


def upload_resume(name: str, content: bytes, mime: str) -> dict[str, Any]:
    """Send the file to the API; raise RuntimeError with a friendly message on failure."""
    try:
        resp = requests.post(
            f"{API_URL}/api/upload", files={"file": (name, content, mime)}, timeout=360
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach the API at {API_URL}. Is the backend running?") from exc
    except requests.Timeout as exc:
        raise RuntimeError("The request timed out. The LLM may be slow; try again.") from exc

    if resp.status_code != 201:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"[{resp.status_code}] {detail}")
    return resp.json()


def chips(items: list[str]) -> str:
    """Render a list of strings as inline code chips."""
    return " ".join(f"`{i}`" for i in items) if items else "_None found_"


def render(record: dict[str, Any]) -> None:
    """Render extracted data as organized cards."""
    data = record["data"]
    contact = data["contact"]

    st.success(f"Extracted from **{record['filename']}**  |  Document ID: `{record['document_id']}`")

    with st.container(border=True):
        st.subheader(f"👤 {contact['name'] or 'Unknown candidate'}")
        cols = st.columns(4)
        for col, (label, key) in zip(
            cols, [("Email", "email"), ("Phone", "phone"), ("Location", "location"), ("LinkedIn", "linkedin")]
        ):
            col.markdown(f"**{label}**\n\n{contact[key] or '—'}")
        extras = [f"{k.title()}: {contact[k]}" for k in ("github", "website") if contact[k]]
        if extras:
            st.caption("  •  ".join(extras))

    if data["summary"]:
        with st.container(border=True):
            st.subheader("📝 Professional Summary")
            st.write(data["summary"])

    with st.container(border=True):
        st.subheader("💼 Work Experience")
        if not data["work_experience"]:
            st.write("_None found_")
        for job in data["work_experience"]:
            st.markdown(f"**{job['role'] or 'Role n/a'}** — {job['company'] or 'Company n/a'}")
            meta = " • ".join(x for x in (job["duration"], job["location"]) if x)
            if meta:
                st.caption(meta)
            for item in job["responsibilities"]:
                st.markdown(f"- {item}")
            st.divider()

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("🎓 Education")
            if not data["education"]:
                st.write("_None found_")
            for edu in data["education"]:
                st.markdown(f"**{edu['degree'] or 'Degree n/a'}**")
                st.caption(" • ".join(x for x in (edu["institution"], edu["year"], edu["grade"]) if x))
    with right:
        with st.container(border=True):
            st.subheader("📜 Certifications")
            if not data["certifications"]:
                st.write("_None found_")
            for cert in data["certifications"]:
                st.markdown(f"**{cert['name']}**")
                st.caption(" • ".join(x for x in (cert["issuer"], cert["year"]) if x))

    with st.container(border=True):
        st.subheader("🛠️ Skills")
        st.markdown("**Technical**")
        st.markdown(chips(data["skills"]["technical"]))
        st.markdown("**Soft skills**")
        st.markdown(chips(data["skills"]["soft"]))

    if data["projects"]:
        with st.container(border=True):
            st.subheader("🚀 Projects")
            for proj in data["projects"]:
                st.markdown(f"**{proj['name']}**")
                st.write(proj["description"])
                if proj["technologies"]:
                    st.markdown(chips(proj["technologies"]))

    if data["languages"]:
        with st.container(border=True):
            st.subheader("🌐 Languages")
            st.markdown(chips(data["languages"]))

    with st.expander("Raw JSON"):
        st.json(record)


st.title("📄 Resume Parser")
st.caption("Upload a PDF or DOCX resume and extract structured information with an LLM.")

uploaded = st.file_uploader("Upload resume", type=["pdf", "docx"])

if st.button("Extract", type="primary", disabled=uploaded is None):
    with st.spinner("Analyzing resume... local models can take a minute."):
        try:
            st.session_state["record"] = upload_resume(
                uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"
            )
        except RuntimeError as err:
            st.session_state.pop("record", None)
            st.error(str(err))

if "record" in st.session_state:
    render(st.session_state["record"])
