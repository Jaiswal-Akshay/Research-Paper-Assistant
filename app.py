import streamlit as st
from langchain_core.documents import Document

from arxiv_ingestion import index_arxiv_topic
from arxiv_rag import answer_arxiv_question
from config import ARXIV_MAX_RESULTS, LLM_MODEL
from vector_store import collection_count


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="arXiv Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session-state initialization
# ============================================================

def initialize_session_state() -> None:
    """
    Create values that must persist across Streamlit reruns.

    Streamlit reruns the script whenever the user interacts with
    a widget, so conversation history must live in session state.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_indexed_topic" not in st.session_state:
        st.session_state.last_indexed_topic = ""

    if "last_indexed_count" not in st.session_state:
        st.session_state.last_indexed_count = 0


initialize_session_state()


# ============================================================
# Source serialization
# ============================================================

def serialize_source(
    document: Document,
) -> dict:
    """
    Convert a LangChain Document into data that can be safely
    retained in Streamlit session state.
    """
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


def extract_abstract(page_content: str) -> str:
    """
    Extract the abstract portion from the formatted document.
    """
    marker = "Abstract:"

    if marker in page_content:
        return page_content.split(
            marker,
            maxsplit=1,
        )[1].strip()

    return page_content.strip()


# ============================================================
# Source display
# ============================================================

def display_sources(
    sources: list[dict],
) -> None:
    """Display retrieved papers and supporting abstracts."""
    if not sources:
        st.info(
            "No supporting papers were retrieved."
        )
        return

    st.markdown("#### Supporting papers")

    for position, source in enumerate(
        sources,
        start=1,
    ):
        metadata = source["metadata"]

        title = metadata.get(
            "title",
            "Untitled paper",
        )

        arxiv_id = metadata.get(
            "arxiv_id",
            "Unknown ID",
        )

        authors = metadata.get(
            "authors",
            "Authors not available",
        )

        published = metadata.get(
            "published",
            "Publication date not available",
        )

        abstract_url = metadata.get(
            "abstract_url",
            "",
        )

        abstract = extract_abstract(
            source["page_content"]
        )

        expander_title = (
            f"{position}. {title} "
            f"(arXiv:{arxiv_id})"
        )

        with st.expander(expander_title):
            st.markdown(
                f"**Authors:** {authors}"
            )

            st.markdown(
                f"**Published:** {published}"
            )

            st.markdown(
                f"**arXiv ID:** `{arxiv_id}`"
            )

            if abstract_url:
                st.markdown(
                    f"[Open paper on arXiv]"
                    f"({abstract_url})"
                )

            st.markdown("**Retrieved abstract**")
            st.write(abstract)


# ============================================================
# Header
# ============================================================

st.title("📚 arXiv Research Assistant")

st.write(
    "Search and index research-paper metadata from arXiv, "
    "then ask questions using local retrieval-augmented "
    "generation."
)

st.info(
    "Answers are based on arXiv titles, authors, publication "
    "metadata, categories and abstracts—not complete paper "
    "content."
)


# ============================================================
# Sidebar: indexing controls
# ============================================================

with st.sidebar:
    st.header("Build the research collection")

    st.write(
        "Enter a focused research topic and choose how many "
        "arXiv papers to retrieve."
    )

    with st.form("arxiv_index_form"):
        topic = st.text_input(
            "Research topic",
            value="retrieval augmented generation",
            help=(
                "Use a focused topic for more relevant "
                "search results."
            ),
        )

        max_results = st.slider(
            "Number of papers",
            min_value=10,
            max_value=ARXIV_MAX_RESULTS,
            value=20,
            step=10,
        )

        index_submitted = st.form_submit_button(
            "Search and index papers",
            type="primary",
            use_container_width=True,
        )

    if index_submitted:
        cleaned_topic = topic.strip()

        if not cleaned_topic:
            st.error(
                "Enter a research topic before indexing."
            )

        else:
            try:
                with st.status(
                    "Building the research collection...",
                    expanded=True,
                ) as status:
                    st.write(
                        "Searching arXiv for relevant papers..."
                    )

                    result = index_arxiv_topic(
                        topic=cleaned_topic,
                        max_results=max_results,
                    )

                    st.write(
                        "Generating local embeddings..."
                    )

                    st.write(
                        "Saving metadata and abstracts "
                        "to Chroma..."
                    )

                    status.update(
                        label=(
                            "Research collection updated"
                        ),
                        state="complete",
                        expanded=False,
                    )

                st.session_state.last_indexed_topic = (
                    cleaned_topic
                )

                st.session_state.last_indexed_count = (
                    result["indexed_count"]
                )

                st.success(
                    f"Indexed "
                    f"{result['indexed_count']} papers."
                )

                if result["retrieved_count"] < max_results:
                    st.warning(
                        "arXiv returned fewer papers than "
                        "requested."
                    )

            except Exception as exc:
                st.error(
                    "The papers could not be indexed."
                )

                st.exception(exc)

    st.divider()

    st.header("Collection status")

    try:
        stored_paper_count = collection_count(
            "arxiv"
        )

        st.metric(
            "Stored papers",
            stored_paper_count,
        )

    except Exception as exc:
        stored_paper_count = 0

        st.warning(
            "The arXiv collection is not available yet."
        )

        with st.expander("Technical details"):
            st.exception(exc)

    if st.session_state.last_indexed_topic:
        st.markdown(
            "**Last indexed topic:**"
        )

        st.write(
            st.session_state.last_indexed_topic
        )

        st.markdown(
            "**Papers processed in last search:**"
        )

        st.write(
            st.session_state.last_indexed_count
        )

    st.divider()

    st.header("Local model")

    st.code(
        LLM_MODEL,
        language=None,
    )

    st.caption(
        "Ollama must be running on this computer."
    )

    if st.button(
        "Clear chat history",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# Existing conversation
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            display_sources(
                message["sources"]
            )


# ============================================================
# Empty-collection guidance
# ============================================================

if stored_paper_count == 0:
    st.warning(
        "The research collection is empty. Use the sidebar "
        "to search and index arXiv papers before asking "
        "questions."
    )


# ============================================================
# Chat input and RAG execution
# ============================================================

question = st.chat_input(
    "Ask a question about the indexed papers...",
    disabled=(stored_paper_count == 0),
)

if question:
    cleaned_question = question.strip()

    if cleaned_question:
        user_message = {
            "role": "user",
            "content": cleaned_question,
        }

        st.session_state.messages.append(
            user_message
        )

        with st.chat_message("user"):
            st.markdown(cleaned_question)

        try:
            with st.chat_message("assistant"):
                with st.spinner(
                    "Retrieving papers and asking "
                    "local Llama 3..."
                ):
                    result = answer_arxiv_question(
                        cleaned_question
                    )

                serialized_sources = [
                    serialize_source(document)
                    for document in result["sources"]
                ]

                st.markdown(result["answer"])

                display_sources(
                    serialized_sources
                )

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": serialized_sources,
            }

            st.session_state.messages.append(
                assistant_message
            )

        except Exception as exc:
            error_message = (
                "I could not process that question. "
                "Confirm that Ollama is running and try again."
            )

            with st.chat_message("assistant"):
                st.error(error_message)

                with st.expander(
                    "Technical details"
                ):
                    st.exception(exc)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )