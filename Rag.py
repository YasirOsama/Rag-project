import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from crewai_tools.tools.base_tool import tool
from crewai import Crew, Task, Agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os

# Securely load sensitive API keys (e.g., Hugging Face API key)
from dotenv import load_dotenv
load_dotenv()

# Initialize Wikipedia API wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Define a custom router tool
@tool
def router_tool(question: str) -> str:
    """Router Function"""
    if 'self-attention' in question:
        return "Discussing self-attention mechanism."
    return "General query."

# Initialize Hugging Face Endpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
api_key = "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"  # Use an environment variable for the API key
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=api_key,
    temperature=0.7,
    timeout=300,
)

# Load and split the PDF document
pdf_path = r'C:\Users\Yasir\Desktop\fun\National Ai policy.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Maximum size of each chunk
    chunk_overlap=20
)
pages = text_splitter.split_documents(docs)

# Initialize FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(pages, embedding_model)

# Define agents
Router_Agent = Agent(
    role='Router',
    goal='Route user question to a vectorstore or web search',
    backstory=(
        "You are an expert at routing a user question to a vectorstore or web search."
        "Use the vectorstore for questions on concepts related to Retrieval-Augmented Generation."
        "Otherwise, use web-search."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Retriever_Agent = Agent(
    role="Retriever",
    goal="Use the information retrieved from the vectorstore to answer the question",
    backstory=(
        "You are an assistant for question-answering tasks."
        "Use the retrieved context to answer the question clearly and concisely."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Answer_Grader_Agent = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You assess whether an answer is useful to resolve a question."
        "If the answer is relevant, generate a clear response."
        "If it is not relevant, perform a web search and return the response."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Define tasks
router_task = Task(
    description=("Analyze the question {question} to determine if it is eligible for a vectorstore or web search."
                 "Return 'vectorstore' or 'websearch' based on the keywords."),
    expected_output="Binary choice: 'websearch' or 'vectorstore'.",
    agent=Router_Agent,
    tools=[router_tool],
)

retriever_task = Task(
    description=("Retrieve information for the question {question} using the appropriate tool based on router_task output."),
    expected_output="A concise and clear answer to the question.",
    agent=Retriever_Agent,
    context=[router_task],
)

answer_task = Task(
    description=("Evaluate the retrieved answer for hallucination and return a clear and concise response."),
    expected_output=("If relevant, return the answer. If not, perform a web search and return a response."),
    context=[retriever_task],
    agent=Answer_Grader_Agent,
)

# Define Crew
rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent, Answer_Grader_Agent],
    tasks=[router_task, retriever_task, answer_task],
    verbose=True,
)

# Streamlit App
st.title("AI Agent for Question Answering")

question = st.text_input("Enter your question:")

if st.button("Submit"):
    if question:
        try:
            response = rag_crew.kickoff(inputs={"question": question})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
