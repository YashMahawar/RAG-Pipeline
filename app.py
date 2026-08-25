import os
import streamlit as st
from dotenv import load_dotenv
from answer import create_prompt, ask_question, get_LLM
from Retrieval_Pipeline import get_vector_db, get_relevantChunks
from Ingestion_Pipeline import complete
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

st.title("Conversational RAG System")

query = st.chat_input("Enter your question")

@st.cache_resource
def init_resources():
    return get_LLM(), get_vector_db()

llm, vector_db = init_resources()


ss = st.session_state
if "chatHistory" not in ss:
    ss.chatHistory = []
if "Processed_files" not in ss:
    ss.Processed_files = []

with st.sidebar:
    st.header("Controls", text_alignment="center")
    file = st.file_uploader("Upload Text File", type=["txt"])
    if st.button("Clear Chat"):
        ss.chatHistory = []
        st.rerun()

if file and file.name not in ss.Processed_files:
    directory = "docs"
    os.makedirs("docs",exist_ok=True)
    path=os.path.join(directory,file.name)

    with open(path,"wb") as f:
        f.write(file.getbuffer())
    complete()
    st.cache_resource.clear()
    ss.Processed_files.append(file.name)
    st.rerun()
# Render custom chat bubbles
for msg in ss.chatHistory:
    is_user = isinstance(msg, HumanMessage)
    
    if is_user:
        # User Bubble: Right-aligned, colored background
        bubble_html = f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
            <div style="background-color: #2b5c8f; color: #ffffff; padding: 10px 16px; border-radius: 16px 16px 2px 16px; max-width: 75%; font-size: 15px; word-wrap: break-word;">
                {msg.content}
            </div>
        </div>
        """
    else:
        # Bot Bubble: Left-aligned, dark/gray background
        bubble_html = f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
            <div style="background-color: #262730; color: #f0f2f6; padding: 10px 16px; border-radius: 16px 16px 16px 2px; max-width: 75%; font-size: 15px; word-wrap: break-word;">
                {msg.content}
            </div>
        </div>
        """
    st.markdown(bubble_html, unsafe_allow_html=True)

if query:
    new_query=ask_question(query,llm,ss.chatHistory)

    relevant_chunks = get_relevantChunks(vector_db,new_query)

    if not relevant_chunks:
        answer = "I dont have enough information to answer your question."

    else:
        message = create_prompt(query,relevant_chunks,ss.chatHistory)
        response = llm.invoke(message)
        answer= response.content

    ss.chatHistory.append(HumanMessage(content = f"{query}"))
    ss.chatHistory.append(AIMessage(content = f"{answer}"))

    ss.chatHistory = ss.chatHistory[-6:]

    st.rerun()

