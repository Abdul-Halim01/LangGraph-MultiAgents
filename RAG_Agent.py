# from dotenv import load_dotenv
# import os
# from typing import TypedDict, Annotated, Sequence
# from operator import add as add_messages

# from langgraph.graph import StateGraph, END
# from langchain_core.messages import (
#     BaseMessage,
#     SystemMessage,
#     HumanMessage,
#     ToolMessage,
# )

# from langchain_google_genai import (
#     ChatGoogleGenerativeAI,
#     GoogleGenerativeAIEmbeddings,
# )

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.tools import tool

# # =========================================================
# # ENV
# # =========================================================
# load_dotenv(override=True)

# # Make sure this exists:
# # export GOOGLE_API_KEY="your-key"
# # or set it in .env

# # =========================================================
# # LLM (Gemini)
# # =========================================================
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,   # deterministic / low hallucination
# )

# # =========================================================
# # Embeddings (Gemini-compatible)
# # =========================================================
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001"
# )

# # =========================================================
# # PDF LOADING
# # =========================================================
# pdf_path = "Stock_Market_Performance_2024.pdf"

# if not os.path.exists(pdf_path):
#     raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# pdf_loader = PyPDFLoader(pdf_path)

# try:
#     pages = pdf_loader.load()
#     print(f"PDF loaded successfully ({len(pages)} pages)")
# except Exception as e:
#     print(f"Error loading PDF: {e}")
#     raise

# # =========================================================
# # CHUNKING
# # =========================================================
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# pages_split = text_splitter.split_documents(pages)

# # =========================================================
# # VECTOR STORE (Chroma)
# # =========================================================
# persist_directory = r"C:\Vaibhav\LangGraph_Book\LangGraphCourse\Agents"
# collection_name = "stock_market"

# os.makedirs(persist_directory, exist_ok=True)

# try:
#     vectorstore = Chroma.from_documents(
#         documents=pages_split,
#         embedding=embeddings,
#         persist_directory=persist_directory,
#         collection_name=collection_name,
#     )
#     print("ChromaDB vector store created!")
# except Exception as e:
#     print(f"Error setting up ChromaDB: {e}")
#     raise

# # =========================================================
# # RETRIEVER
# # =========================================================
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}
# )

# @tool
# def retriever_tool(query: str) -> str:
#     """
#     Search Stock Market Performance 2024 document.
#     """
#     docs = retriever.invoke(query)

#     if not docs:
#         return "No relevant information found in the document."

#     results = []
#     for i, doc in enumerate(docs):
#         results.append(f"Document {i+1}:\n{doc.page_content}")

#     return "\n\n".join(results)

# tools = [retriever_tool]
# llm = llm.bind_tools(tools)

# # =========================================================
# # AGENT STATE
# # =========================================================
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# def should_continue(state: AgentState):
#     last_message = state["messages"][-1]
#     return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

# # =========================================================
# # SYSTEM PROMPT
# # =========================================================
# system_prompt = """
# You are an intelligent AI assistant who answers questions about
# Stock Market Performance in 2024 using ONLY the provided PDF.

# Use the retriever tool whenever factual information is required.
# Always cite relevant document excerpts in your answers.
# """

# tools_dict = {tool.name: tool for tool in tools}

# # =========================================================
# # LLM NODE
# # =========================================================
# def call_llm(state: AgentState) -> AgentState:
#     messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
#     response = llm.invoke(messages)
#     return {"messages": [response]}

# # =========================================================
# # TOOL NODE
# # =========================================================
# def take_action(state: AgentState) -> AgentState:
#     tool_calls = state["messages"][-1].tool_calls
#     results = []

#     for call in tool_calls:
#         name = call["name"]
#         query = call["args"].get("query", "")

#         print(f"Calling tool: {name} | Query: {query}")

#         if name not in tools_dict:
#             output = "Invalid tool name."
#         else:
#             output = tools_dict[name].invoke(query)

#         results.append(
#             ToolMessage(
#                 tool_call_id=call["id"],
#                 name=name,
#                 content=str(output),
#             )
#         )

#     return {"messages": results}

# # =========================================================
# # GRAPH
# # =========================================================
# graph = StateGraph(AgentState)

# graph.add_node("llm", call_llm)
# graph.add_node("retriever_agent", take_action)

# graph.add_conditional_edges(
#     "llm",
#     should_continue,
#     {True: "retriever_agent", False: END},
# )

# graph.add_edge("retriever_agent", "llm")
# graph.set_entry_point("llm")

# rag_agent = graph.compile()

# # =========================================================
# # RUN LOOP
# # =========================================================
# def running_agent():
#     print("\n=== RAG AGENT (Gemini) ===")

#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in {"exit", "quit"}:
#             break

#         result = rag_agent.invoke(
#             {"messages": [HumanMessage(content=user_input)]}
#         )

#         print("\n=== ANSWER ===")
#         print(result["messages"][-1].content)

# running_agent()



from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

# ======================================================
# ENV
# ======================================================
load_dotenv()

# ======================================================
# LLM (Gemini – FREE TIER)
# ======================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# ======================================================
# EMBEDDINGS (Gemini – FREE)
# ======================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ======================================================
# LOAD PDF
# ======================================================
PDF_PATH = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"Missing PDF: {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Loaded {len(pages)} pages")

# ======================================================
# SPLIT DOCUMENTS
# ======================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = splitter.split_documents(pages)

# ======================================================
# CHROMA (LOCAL, FREE)
# ======================================================
PERSIST_DIR = "./chroma_db"
COLLECTION = "stock_market"

if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# ======================================================
# TOOL
# ======================================================
@tool
def search_stock_pdf(query: str) -> str:
    """Search Stock Market Performance 2024 PDF"""
    results = retriever.invoke(query)

    if not results:
        return "No relevant information found."

    return "\n\n".join(
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )

tools = [search_stock_pdf]
llm = llm.bind_tools(tools)

# ======================================================
# AGENT STATE
# ======================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return bool(getattr(last, "tool_calls", []))

# ======================================================
# SYSTEM PROMPT
# ======================================================
SYSTEM_PROMPT = """
You are a RAG assistant answering questions ONLY using
the Stock Market Performance 2024 PDF.

Always use the search tool before answering.
Cite information clearly from the documents.
"""

tool_map = {t.name: t for t in tools}

# ======================================================
# LLM NODE
# ======================================================
def call_llm(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

# ======================================================
# TOOL NODE
# ======================================================
def run_tools(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    outputs = []

    for call in tool_calls:
        name = call["name"]
        query = call["args"].get("query", "")

        print(f"🔧 Tool call → {name}: {query}")

        result = tool_map[name].invoke(query)

        outputs.append(
            ToolMessage(
                tool_call_id=call["id"],
                name=name,
                content=str(result),
            )
        )

    return {"messages": outputs}

# ======================================================
# GRAPH
# ======================================================
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("tools", run_tools)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "tools", False: END},
)

graph.add_edge("tools", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# ======================================================
# RUN
# ======================================================
def run():
    print("\n=== GEMINI + CHROMA RAG (FREE) ===")

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() in {"exit", "quit"}:
            break

        result = rag_agent.invoke(
            {"messages": [HumanMessage(content=q)]}
        )

        print("\n=== ANSWER ===")
        final_msg = result["messages"][-1].content

        if isinstance(final_msg, list):
            print(final_msg[0]["text"])
        else:
            print(final_msg)


run()
