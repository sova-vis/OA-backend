"""
Step 6: Chat interface.

A simple Streamlit chat UI over the RAG pipeline built in Steps 1-5:
parse -> chunk -> embed -> retrieve -> generate.

Run:
    venv\\Scripts\\python.exe -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from drive_pdf import download_pdf_bytes, render_page_image  # noqa: E402
from generate_answer import DRIVE_FILE_MAP, _normalize_path, generate, question_preview  # noqa: E402
from pdf_server import start_pdf_server  # noqa: E402
from retrieve import Retriever  # noqa: E402

st.set_page_config(page_title="O Level Past Paper Chatbot", page_icon="📚")


@st.cache_resource
def get_retriever():
    start_pdf_server()
    return Retriever()


@st.cache_data
def get_subjects():
    r = get_retriever()
    metas = r.collection.get(limit=100000, include=["metadatas"])["metadatas"]
    return sorted({m["subject"] for m in metas if m.get("subject")})


@st.cache_data(show_spinner=False, max_entries=64)
def cached_pdf_bytes(file_id: str) -> bytes:
    """Downloaded once per paper per server run, then reused for every
    question anyone looks up out of that same paper (and across users, since
    st.cache_data is shared server-side, not per-session)."""
    return download_pdf_bytes(file_id)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_page_image(file_id: str, page: int) -> bytes:
    return render_page_image(cached_pdf_bytes(file_id), page)


st.title("📚 O Level Past Paper Chatbot")

mode_label = st.radio(
    "Mode",
    options=["Ask", "Find"],
    horizontal=True,
    label_visibility="collapsed",
    help=(
        "Ask: get an explanation of a concept, grounded in the papers and the web.\n\n"
        "Find: look up exactly which years/papers a topic was asked in - papers only, no web search."
    ),
)
mode = "ask" if mode_label == "Ask" else "find"

if mode == "ask":
    st.caption(
        "**Ask mode** — explain a concept using the past papers and the web. "
        "Not for \"which years was X asked\" — switch to Find for that."
    )
else:
    st.caption(
        "**Find mode** — look up which years/papers a topic was asked in, from the past papers only. "
        "No web search, no explanations."
    )

with st.sidebar:
    st.header("Filter (optional)")
    subjects = ["All subjects"] + get_subjects()
    selected_subject = st.selectbox("Subject", subjects)
    subject_filter = None if selected_subject == "All subjects" else selected_subject

    st.divider()
    st.markdown(
        "**Ask examples:**\n"
        "- Explain how photosynthesis works\n"
        "- Explain the difference between fixed and variable costs\n\n"
        "**Find examples:**\n"
        "- Which years was depreciation asked in Accounting?\n"
        "- How many times has supply and demand come up in Economics?"
    )

def render_occurrences(occurrences: list, widget_key: str):
    """Interactive replacement for the plain markdown links table: a
    checkbox at the start of each row acts as the link - selecting it opens
    that exact PDF page, rendered as an image right below the table.
    Rendering the page as an image rather than embedding a PDF viewer
    sidesteps a confirmed issue: neither Drive's own viewer nor Google Docs
    Viewer honors a "#page=N" jump when opened as a direct link - both
    stayed on page 1 regardless of the fragment - so there's no "jump" here
    that can fail, the requested page is just what gets drawn."""
    if not occurrences:
        return

    event = st.dataframe(
        [
            {
                "Year": m.get("year"),
                "Session": (m.get("session") or "").replace("_", "/"),
                "Paper": m.get("paper"),
                "Variant": m.get("variant") if m.get("variant") is not None else "-",
                "Q#": m.get("question_number"),
                "Question": question_preview(m.get("question_text", "")),
            }
            for m in occurrences
        ],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"select_{widget_key}",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        m = occurrences[selected_rows[0]]
        drive_id = DRIVE_FILE_MAP.get(_normalize_path(m.get("source_file", "")))
        if not drive_id:
            st.warning("This paper isn't mapped to Drive yet, so its exact page can't be shown here.")
            return
        with st.spinner("Loading page..."):
            try:
                image_bytes = cached_page_image(drive_id, m.get("page", 1))
            except Exception as e:
                st.error(f"Couldn't load that page: {e}")
                return
        st.image(image_bytes, use_container_width=True)


if "messages_by_mode" not in st.session_state:
    st.session_state.messages_by_mode = {"ask": [], "find": []}

history = st.session_state.messages_by_mode[mode]

for i, msg in enumerate(history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("occurrences"):
            render_occurrences(msg["occurrences"], widget_key=f"{mode}_{i}")

if prompt := st.chat_input("Ask a question about O Level past papers..."):
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_text = (
            "Searching the papers and the web..." if mode == "ask" else "Searching past papers..."
        )
        with st.spinner(spinner_text):
            try:
                answer, result = generate(
                    prompt, subject=subject_filter, mode=mode, include_table=False
                )
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                result = None
        st.markdown(answer)
        occurrences = result.get("occurrences") if result else None
        if occurrences:
            render_occurrences(occurrences, widget_key=f"{mode}_{len(history)}")
        if result:
            st.caption(f"Mode: `{mode_label}`")

    history.append(
        {
            "role": "assistant",
            "content": answer,
            "occurrences": result.get("occurrences") if result else None,
        }
    )
