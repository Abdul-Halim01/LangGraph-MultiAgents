import pandas as pd

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import os
from dotenv import load_dotenv


# === LLM setup
load_dotenv(override=True)
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=2,
    temperature=0.1,
    max_tokens=500,
)

# === Sample Data ===
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'sales': [1_000_000, 120, 115, 140, 160, 155, 180, 190, 185, 200],
    'profit': [20, 25, 23, 30, 35, 33, 40, 42, 41, 45]
})

# === Tools ===
@tool
def execute_python(code: str) -> str:
    """Execute Python code to analyze the DataFrame 'df'. 
    Variables available: df (pandas DataFrame), pd (pandas module).
    Store your result in a variable called 'result' to return it.
    """
    try:
        local_vars = {"df": df, "pd": pd}
        exec(code, {"__builtins__": __builtins__, "pd": pd}, local_vars)
        return str(local_vars.get("result", "Code executed. Use 'result' variable."))
    except Exception as e:
        return f"Error: {str(e)}"


# === Create Agent ===
tools = [execute_python]
llm_with_tools = model.bind_tools(tools)

def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: MessagesState):
    """Route to tools or end"""
    return "tools" if state["messages"][-1].tool_calls else END

# === Build Graph ===
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tools", END])
graph.add_edge("tools", "agent")
# With LangGraph API, persistence is handled automatically by the platform.
# agent = graph.compile(checkpointer=MemorySaver())
agent = graph.compile()

# === Standalone Execution ===
if __name__ == "__main__":    
    system_msg = SystemMessage(content=f"""
        You are an expert data scientist. 
        When asked a question or given a task, you will create and run code to analyze the provided data provided in a pandas DataFrame called 'df'.            
        Here is the DataFrame schema: {df.dtypes.to_dict()}.
        Here is some of the data: {df.head().to_string()}.
        Create the code, execute it, and provide the answer."""
    )
      
    thread_id = 0
    config = {"configurable": {"thread_id": thread_id+1}}

    prompt = "What are the average sales?"
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    print("\n" + "="*60)
    print("AGENT RESPONSE:")
    print("="*60)
    print(result['messages'][-1].content)