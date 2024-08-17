import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
from langchain.docstore.document import Document  # Import Document class

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit app configuration
st.set_page_config(page_title="Summarize", page_icon="📄")

# Title with a container
st.markdown(
    """
    <style>
    .title-container {
        position: relative;
        background-image: url("https://imgs.search.brave.com/LMbtcRP9xh_GYr28J_aW054OB4mUYjDtd8Gu6vlneYM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/ZnJlZS1waG90by9h/YnN0cmFjdC1iYWNr/Z3JvdW5kLXdpdGgt/cmVkLWxpbmVzXzEz/NjEtMzUzMS5qcGc_/c2l6ZT02MjYmZXh0/PWpwZw");
        background-size: cover;
        padding: 50px;
        border-radius: 15px;
        text-align: center;
        overflow: hidden;
        transition: transform 0.5s ease;
    }
    
    .title-container h1, .title-container h2 {
        color: white;
        font-family: 'Quicksand', sans-serif;
        position: relative;
        z-index: 2;
        transition: transform 0.3s ease, text-shadow 0.3s ease;
    }

    .title-container h1 {
        font-size: 3em;
    }

    .title-container h2 {
        font-size: 1.5em;
    }

    .title-container:hover h1, .title-container:hover h2 {
        transform: scale(1.1) perspective(500px) rotateX(5deg);
        text-shadow: 3px 3px 5px rgba(0, 0, 0, 0.3);
    }

    .title-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.2);
        z-index: 1;
        transition: left 0.5s ease-in-out;
    }

    .title-container:hover::before {
        left: 100%;
    }
    </style>
    <div class="title-container">
        <h1>📄 Summarize</h1>
        <h2>Summarize Text From YouTube or Website</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Input URL box
generic_url = st.text_input("Enter a URL", "")

# Load Summarization LLM with Groq
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Initialize session state for transcript and summary
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Fetch and display transcript
if st.button("Show Transcript"):
    if generic_url and validators.url(generic_url):
        try:
            with st.spinner("Fetching transcript..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                
                docs = loader.load()
                if docs:
                    # Ensure the content is in plain text
                    transcript_text = docs[0].page_content if hasattr(docs[0], 'page_content') else str(docs[0])
                    st.session_state.transcript = transcript_text
                    st.text_area("Transcript", value=st.session_state.transcript, height=300)
                else:
                    st.error("No content found.")
        except Exception as e:
            st.error(f"An error occurred while fetching the transcript: {e}")
    else:
        st.error("Please enter a valid URL.")

# Summarize the content
if st.button("Summarize"):
    if st.session_state.transcript:
        try:
            with st.spinner("Summarizing content..."):
                # Convert the transcript string to a Document object
                docs = [Document(page_content=st.session_state.transcript)]
                
                prompt_template = PromptTemplate(template="Summarize this content in 300 words:\n{text}", input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
                summary = chain.run(docs)
                
                st.session_state.summary = summary
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred while summarizing: {e}")
    else:
        st.error("Transcript is not available. Please fetch the transcript first by clicking 'Show Transcript'.")
    