import os
import streamlit as st
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import extract_video_id, get_transcript, process_transcript

from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client
def init_groq():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2
    )

# Format retrieved documents
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Create the QA chain
def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    prompt = PromptTemplate(
        template="""
        You are a helpful and honest YouTube assistant. Use **only** the information provided in the transcript context below to answer the question.

        - If the context does not contain enough information to answer the question, respond with:
        "This topic is not related to the content of the video."

        Context:
        {context}

        Question: {question}

        Answer:
        """,
        input_variables=['context', 'question']
    )
    
    return RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    }) | prompt | init_groq() | StrOutputParser()

# Streamlit UI
def main():
    st.set_page_config(page_title="YouTube Chatbot", page_icon="▶️")
    st.title("YouTube Video Chatbot")
    
    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None
    if "thumbnail_url" not in st.session_state:
        st.session_state.thumbnail_url = None
    
    # Main interface for video input
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_url = st.text_input("YouTube URL:", placeholder="Enter YouTube video URL", key="url_input")
    with col2:
        language = st.selectbox("Language", ["en", "mr", "fr", "es", "de", "hi", "ja", "ru", "zh"], index=0)
    
    # Reset button
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.video_processed = False
        st.session_state.current_video_id = None
        st.session_state.thumbnail_url = None
    
    # Display thumbnail if available
    if st.session_state.thumbnail_url:
        st.image(st.session_state.thumbnail_url, caption="Video Thumbnail", use_container_width=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
            return
        
        # Process video only if it's new or not processed yet
        if not st.session_state.video_processed or st.session_state.current_video_id != video_id:
            st.session_state.current_video_id = video_id
            
            # Store thumbnail URL in session state
            st.session_state.thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            
            # Display thumbnail immediately
            st.image(st.session_state.thumbnail_url, caption="Video Thumbnail", use_container_width=True)
            
            with st.spinner("Processing video..."):
                transcript = get_transcript(video_id, language)
                
                if not transcript:
                    st.error("Transcript not available for this video.")
                    return
                
                vector_store = process_transcript(transcript)
                qa_chain = create_qa_chain(vector_store)
                st.session_state.qa_chain = qa_chain
                st.session_state.video_processed = True
                st.session_state.messages = []  # Clear previous chat when loading new video
                st.success("Video processed! Ask questions below.")
        
        # Chat input
        if prompt := st.chat_input("Ask about the video..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()