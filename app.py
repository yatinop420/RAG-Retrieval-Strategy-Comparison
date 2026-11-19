
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
import time

from retrievers import (
    similarity_search,
    mmr_search,
    multi_query_search,
    bm25_search,
    hybrid_search
)

from generator import generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Ask Your AI",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# RETRIEVAL METHODS
# --------------------------------------------------

retrievers = {
    "Similarity Search": similarity_search,
    "MMR Search": mmr_search,
    "Multi-Query": multi_query_search,
    "BM25": bm25_search,
    "Hybrid": hybrid_search
}


# --------------------------------------------------
# HELPER FUNCTION: DISPLAY SOURCES
# --------------------------------------------------

def display_sources(documents):

    if not documents:
        st.info("No relevant sources were retrieved.")
        return

    with st.expander("View retrieved sources"):

        for index, doc in enumerate(documents, start=1):

            page = doc.metadata.get(
                "page",
                doc.metadata.get("page_number", "Unknown")
            )

            st.markdown(f"**Source {index} — Page {page}**")

            st.write(doc.page_content)

            if index < len(documents):
                st.divider()


# --------------------------------------------------
# HELPER FUNCTION: GENERATE ANSWER
# --------------------------------------------------

def process_question(question, method):

    retriever = retrievers[method]

    # Retrieval
    retrieval_start = time.perf_counter()

    documents = retriever(question, k=4)

    retrieval_time = time.perf_counter() - retrieval_start

    # Generation
    generation_start = time.perf_counter()

    answer = generate_answer(question, documents)

    generation_time = time.perf_counter() - generation_start

    return {
        "answer": answer,
        "documents": documents,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("Customize your search experience.")

question = st.sidebar.text_area(
    "Ask your question",
    placeholder="What would you like to know?",
    height=120
)

mode = st.sidebar.radio(
    "Select Mode",
    ["Quick Answer", "Compare All"]
)

selected_method = st.sidebar.selectbox(
    "Retrieval Method",
    list(retrievers.keys()),
    index=4
)

submit = st.sidebar.button(
    "Generate Answer",
    use_container_width=True,
    type="primary"
)


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🤖 Ask Your AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Your intelligent document-powered assistant</p>',
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown("### Welcome!")

st.write(
    "Ask questions about your uploaded knowledge base "
    "and get AI-generated answers grounded in relevant documents."
)


# --------------------------------------------------
# QUICK ANSWER MODE
# --------------------------------------------------

if submit:

    if not question.strip():

        st.warning("Please enter a question first.")

    else:

        if mode == "Quick Answer":

            st.markdown("## Answer")

            st.caption(f"Retrieval Method: {selected_method}")

            try:

                with st.spinner("Searching documents and generating answer..."):

                    result = process_question(
                        question,
                        selected_method
                    )

                st.markdown("### Response")

                st.markdown(result["answer"])

                st.divider()

                col1, col2 = st.columns(2)

                col1.metric(
                    "Retrieval Time",
                    f"{result['retrieval_time']:.2f} sec"
                )

                col2.metric(
                    "Generation Time",
                    f"{result['generation_time']:.2f} sec"
                )

                display_sources(result["documents"])

            except Exception as e:

                st.error(f"Something went wrong: {e}")


        # --------------------------------------------------
        # COMPARE ALL MODE
        # --------------------------------------------------

        elif mode == "Compare All":

            st.markdown("## Compare All Retrieval Methods")

            st.caption(
                "The same question will be processed using all five methods."
            )

            for method in retrievers:

                st.markdown(f"### {method}")

                try:

                    with st.spinner(f"Running {method}..."):

                        result = process_question(
                            question,
                            method
                        )

                    st.markdown("**Generated Answer**")

                    st.markdown(result["answer"])

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Retrieval Time",
                        f"{result['retrieval_time']:.2f} sec"
                    )

                    col2.metric(
                        "Generation Time",
                        f"{result['generation_time']:.2f} sec"
                    )

                    display_sources(result["documents"])

                except Exception as e:

                    st.error(f"{method} failed: {e}")

                st.divider()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Built with Python, LangChain, ChromaDB, Groq and Streamlit."
)
