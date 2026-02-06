# LangGraph MultiAgents

A comprehensive implementation of intelligent multi-agent systems using LangGraph and Google's Gemini AI. This project demonstrates various agent architectures including ReAct agents, RAG (Retrieval-Augmented Generation) agents, conversational agents, document drafting agents, and a supervisor-based multi-agent orchestration system.

![LangGraph](https://img.shields.io/badge/LangGraph-Framework-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Gemini](https://img.shields.io/badge/Google-Gemini_AI-orange)
![UV](https://img.shields.io/badge/UV-Package_Manager-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Agents Breakdown](#agents-breakdown)
  - [1. Data Analysis Agent (agent.py)](#1-data-analysis-agent-agentpy)
  - [2. Drafter Agent (Drafter.py)](#2-drafter-agent-drafterpy)
  - [3. Lab Agent (Lab_agent.py)](#3-lab-agent-lab_agentpy)
  - [4. RAG Agent (RAG_Agent.py)](#4-rag-agent-rag_agentpy)
  - [5. ReAct Agent (ReAct_Agent.py)](#5-react-agent-react_agentpy)
  - [6. Supervisor Multi-Agent System (Supervisor.ipynb)](#6-supervisor-multi-agent-system-supervisoripynb)
- [Installation](#installation)
- [UV Package Manager](#uv-package-manager)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)

---

## ⚡ Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/Abdul-Halim01/LangGraph-MultiAgents.git
cd LangGraph-MultiAgents

# 2. Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# OR: pip install uv

# 3. Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. Install dependencies
uv pip install -e .

# 5. Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 6. Run an agent
python agent.py
```

**Get your Gemini API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 🎯 Overview

This project showcases the power of LangGraph for building sophisticated multi-agent AI systems. Each agent is designed with a specific purpose and demonstrates different aspects of agent architectures:

- **Tool-based agents** that can execute code and manipulate data
- **Conversational agents** with memory and context management
- **RAG agents** that retrieve and synthesize information from documents
- **Document manipulation agents** with specialized workflows
- **Supervisor systems** that orchestrate multiple specialized agents

All agents leverage **Google's Gemini AI** models for natural language understanding and generation, showcasing enterprise-grade AI capabilities with cost-effective solutions.

---

## 🏗️ Architecture

The project implements several agent patterns:

```
┌─────────────────────────────────────────────────────────────┐
│                    LANGGRAPH MULTIAGENTS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Analysis│  │   Drafter    │  │  Lab Agent   │      │
│  │    Agent     │  │    Agent     │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  RAG Agent   │  │ ReAct Agent  │  │  Supervisor  │      │
│  │              │  │              │  │   System     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns Implemented:

1. **ReAct Pattern** (Reasoning + Acting)
2. **RAG Pattern** (Retrieval-Augmented Generation)
3. **State Management** with LangGraph StateGraph
4. **Tool Integration** with conditional routing
5. **Multi-Agent Orchestration** with supervisor pattern
6. **Quality Validation** with validator agent

### Supervisor System Architecture:

```
                         __start__
                             │
                             ▼
                      ┌──────────────┐
                      │  Supervisor  │◄──┐
                      └──────┬───────┘   │
                             │           │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐     ┌──────────┐
    │  Coder   │      │ Enhancer │     │Researcher│
    └────┬─────┘      └────┬─────┘     └────┬─────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                    ┌──────────────┐
                    │  Validator   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   __end__    │
                    └──────────────┘
                    
Dotted lines (···) = Conditional routing
Solid lines (───) = Direct edges
```

---

## 📁 Project Structure

```
LangGraph-MultiAgents/
│
├── agent.py                    # Data Analysis Agent with Python execution
├── Drafter.py                  # Document creation and editing agent
├── Lab_agent.py                # Conversational agent with memory
├── RAG_Agent.py                # RAG agent for PDF document retrieval
├── ReAct_Agent.py              # ReAct pattern implementation
├── Supervisor.ipynb            # Multi-agent supervisor orchestration
├── langgraph.json              # LangGraph configuration
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Dependency lock file
├── .python-version             # Python version specification
├── Stock_Market_Performance_2024.pdf  # Sample document for RAG
└── README.md                   # This file
```

---

## 🤖 Agents Breakdown

### 1. Data Analysis Agent (`agent.py`)

**Purpose**: Execute Python code to analyze pandas DataFrames dynamically.

**Key Features**:
- Python code execution capability via custom tool
- Automatic data analysis from natural language queries
- Safe code execution environment
- Integration with pandas for data manipulation

**Architecture**:
```python
StateGraph Flow:
START → agent → [has tool_calls?] → tools → agent → END
                      ↓ no
                     END
```

**How it Works**:

1. **LLM Setup**: Uses Google's Gemini 2.5 Flash model with low temperature (0.1) for deterministic responses
   ```python
   model = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash",
       temperature=0.1,
       max_tokens=500,
   )
   ```

2. **Sample Data**: Creates a pandas DataFrame with sales and profit data
   ```python
   df = pd.DataFrame({
       'date': pd.date_range('2024-01-01', periods=10),
       'sales': [1_000_000, 120, 115, 140, 160, 155, 180, 190, 185, 200],
       'profit': [20, 25, 23, 30, 35, 33, 40, 42, 41, 45]
   })
   ```

3. **Python Execution Tool**: 
   - Executes arbitrary Python code with access to `df` and `pd`
   - Expects results in a `result` variable
   - Safe execution with exception handling
   ```python
   @tool
   def execute_python(code: str) -> str:
       """Execute Python code to analyze DataFrame 'df'"""
       local_vars = {"df": df, "pd": pd}
       exec(code, {"__builtins__": __builtins__, "pd": pd}, local_vars)
       return str(local_vars.get("result", "Code executed"))
   ```

4. **Graph Logic**:
   - Agent node calls the LLM with tool binding
   - Conditional routing checks for tool calls
   - Tools node executes Python code
   - Loops back to agent for response synthesis

**Use Cases**:
- Automated data analysis from natural language
- Quick statistical computations
- Data exploration and insights generation
- Business intelligence queries

**Example Interaction**:
```
User: "What are the average sales?"
Agent: [Generates code] → [Executes: df['sales'].mean()] → "The average sales are 1,100,234.5"
```

---

### 2. Drafter Agent (`Drafter.py`)

**Purpose**: Interactive document creation and editing with tool-based workflow.

**Key Features**:
- Document content management via global state
- Update and save operations as tools
- Interactive CLI interface
- Conditional workflow termination
- Real-time document state tracking

**Architecture**:
```python
StateGraph Flow:
agent → tools → [saved?] → END
         ↓ no
       agent (loop)
```

**How it Works**:

1. **Global Document State**:
   ```python
   document_content = ""  # Stores the current document
   ```

2. **Tools**:
   - **`update`**: Updates the entire document content
     ```python
     @tool
     def update(content: str) -> str:
         global document_content
         document_content = content
         return f"Document updated: {document_content}"
     ```
   
   - **`save`**: Saves document to a text file and triggers workflow termination
     ```python
     @tool
     def save(filename: str) -> str:
         # Ensures .txt extension
         # Writes document_content to file
         # Returns success message
     ```

3. **Agent State Management**:
   ```python
   class AgentState(TypedDict):
       messages: Annotated[Sequence[BaseMessage], add_messages]
   ```

4. **Interactive Loop**:
   - Agent prompts user for input
   - LLM decides which tool to use based on user intent
   - System prompt guides the agent's behavior
   - Tools execute and return results
   - Conditional edge checks if document was saved (termination condition)

5. **Conditional Termination Logic**:
   ```python
   def should_continue(state: AgentState) -> str:
       # Checks if most recent ToolMessage contains "saved" and "document"
       # Returns "end" to terminate, "continue" to loop
   ```

**Workflow**:
1. User starts the agent
2. Agent asks what to create/modify
3. User provides instructions
4. Agent uses `update` tool to modify document
5. User can continue editing or save
6. Agent uses `save` tool → workflow ends

**Use Cases**:
- Interactive document drafting
- Content creation with iterative refinement
- Automated report generation
- Note-taking and documentation

**Example Interaction**:
```
USER: Create a meeting summary
AI: [Uses update tool] Document updated with meeting summary
USER: Save it as "meeting_notes"
AI: [Uses save tool] Document saved to meeting_notes.txt
→ Workflow ends
```

---

### 3. Lab Agent (`Lab_agent.py`)

**Purpose**: Simple conversational agent with persistent conversation history.

**Key Features**:
- Conversation memory across interactions
- Message history persistence to file
- Clean state management
- Interactive CLI interface

**Architecture**:
```python
StateGraph Flow (per message):
START → process → END
```

**How it Works**:

1. **State Definition**:
   ```python
   class AgentState(TypedDict):
       messages: List[Union[HumanMessage, AIMessage]]
   ```

2. **LLM Configuration**:
   - Uses Gemini 2.5 Flash Lite (lightweight, fast)
   - Temperature 0.1 for consistent responses
   - 500 token limit

3. **Processing Node**:
   ```python
   def process(state: AgentState) -> AgentState:
       response = llm.invoke(state["messages"])
       state["messages"].append(AIMessage(content=response.content))
       return state
   ```

4. **Conversation Loop**:
   - Maintains `conversation_history` list
   - Each turn appends HumanMessage and AIMessage
   - State is passed through the graph for each interaction
   - History persists across all interactions

5. **Persistence**:
   - Saves entire conversation to `logging.txt`
   - Formats messages by type (Human/AI)
   - UTF-8 encoding for special characters

**Workflow**:
1. User enters a message
2. Message added to conversation history
3. History passed to LLM via graph
4. AI response added to history
5. Loop continues until user types "exit"
6. Full conversation saved to file

**Use Cases**:
- Simple chatbot interfaces
- Conversation logging and analysis
- Testing conversational flows
- Customer support simulations

**Example Conversation Flow**:
```
#1
User: Hello, how are you?
AI: I'm doing well, thank you! How can I help you today?

#2
User: Tell me about Python
AI: Python is a high-level programming language...

exit → Saves to logging.txt
```

---

### 4. RAG Agent (`RAG_Agent.py`)

**Purpose**: Retrieval-Augmented Generation for question-answering from PDF documents.

**Key Features**:
- PDF document loading and processing
- Vector embeddings for semantic search
- ChromaDB for vector storage
- Context-aware question answering
- Free-tier components (Gemini + HuggingFace embeddings)

**Architecture**:
```python
StateGraph Flow:
llm → [has tool_calls?] → tools → llm → [no tool_calls?] → END
         ↓ no
        END
```

**How it Works**:

1. **Document Processing Pipeline**:
   ```python
   # Load PDF
   loader = PyPDFLoader("Stock_Market_Performance_2024.pdf")
   pages = loader.load()
   
   # Split into chunks
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=200,  # Maintains context between chunks
   )
   docs = splitter.split_documents(pages)
   ```

2. **Embeddings**:
   - Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`
   - Free, lightweight, and effective for semantic search
   - Creates vector representations of text chunks

3. **Vector Store (ChromaDB)**:
   ```python
   vectorstore = Chroma.from_documents(
       documents=docs,
       embedding=embeddings,
       persist_directory="./chroma_db",
       collection_name="stock_market",
   )
   ```
   - Persists to disk (reusable across sessions)
   - Enables similarity search
   - Returns top-k most relevant chunks

4. **Retrieval Tool**:
   ```python
   @tool
   def search_stock_pdf(query: str) -> str:
       """Search Stock Market Performance 2024 PDF"""
       results = retriever.invoke(query)  # Returns top 5 similar chunks
       return formatted_results
   ```

5. **Agent Workflow**:
   - User asks a question
   - LLM receives system prompt emphasizing tool use
   - LLM generates tool call with search query
   - Tool retrieves relevant document chunks
   - LLM synthesizes answer from retrieved context
   - Cites sources from documents

6. **System Prompt**:
   ```python
   SYSTEM_PROMPT = """
   You are a RAG assistant answering questions ONLY using
   the Stock Market Performance 2024 PDF.
   
   Always use the search tool before answering.
   Cite information clearly from the documents.
   """
   ```

**RAG Pipeline**:
```
Question → LLM (tool call) → Retrieve chunks → LLM (synthesis) → Answer
```

**Use Cases**:
- Document Q&A systems
- Knowledge base queries
- Research assistance
- Compliance and policy questions

**Example Interaction**:
```
User: "What was the stock market performance in Q1 2024?"
Agent: [Calls search_stock_pdf tool]
       [Retrieves relevant chunks about Q1 performance]
       [Synthesizes answer]
       "According to the document, Q1 2024 showed strong performance with..."
```

---

### 5. ReAct Agent (`ReAct_Agent.py`)

**Purpose**: Implements the ReAct (Reasoning + Acting) pattern with tool integration.

**Key Features**:
- ReAct pattern implementation
- Tool calling with explanations
- Conditional routing based on tool usage
- System-prompted reasoning

**Architecture**:
```python
StateGraph Flow:
our_model → [has tool_calls?] → tools → our_model → END
               ↓ no
              END
```

**How it Works**:

1. **State Definition**:
   ```python
   class AgentState(TypedDict):
       messages: Annotated[Sequence[BaseMessage], add_messages]
   ```
   - Uses `add_messages` reducer to append messages automatically

2. **Tool Definition**:
   ```python
   @tool
   def add(a: int, b: int) -> str:
       """Addition function for 2 integers"""
       return f"Add tool return {a+b}"
   ```

3. **LLM with Tools**:
   ```python
   llm = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash-lite",
       max_tokens=200,
       temperature=0.1,
   )
   llm_with_tools = llm.bind_tools(tools)
   ```

4. **Model Call Node**:
   ```python
   def model_call(state: AgentState) -> AgentState:
       SystemPrompt = SystemMessage(
           content="You are Alpo. Add explanation for your answer."
       )
       response = llm_with_tools.invoke([SystemPrompt] + state["messages"])
       return {"messages": [response]}
   ```

5. **Conditional Routing**:
   ```python
   def should_continue(state: AgentState):
       last_message = state['messages'][-1]
       if last_message.tool_calls:
           return "continue"  # Route to tools
       else:
           return "end"       # Terminate
   ```

6. **ReAct Pattern**:
   - **Reasoning**: LLM analyzes the query
   - **Acting**: LLM decides to use a tool
   - **Observing**: Tool executes and returns result
   - **Reasoning**: LLM incorporates result and explains

7. **Pretty Streaming**:
   ```python
   def print_stream(stream):
       for s in stream:
           message = s["messages"][-1]
           message.pretty_print()  # Formats output nicely
   ```

**ReAct Flow**:
```
User Query → LLM (Reason) → Tool Call (Act) → Execute Tool (Observe) 
→ LLM (Reason with result) → Explain Answer
```

**Use Cases**:
- Mathematical computations
- Multi-step reasoning tasks
- Tool-augmented problem solving
- Interactive calculators

**Example Interaction**:
```
User: "Add 34 + 21 + 7"
AI (Reasoning): "I need to add these numbers together"
AI (Acting): [Calls add tool with appropriate arguments]
Tool: Returns "Add tool return 62"
AI (Reasoning + Explaining): "The sum is 62. I used the addition tool 
because it provides accurate arithmetic computation."
```

---

### 6. Supervisor Multi-Agent System (`Supervisor.ipynb`)

**Purpose**: Orchestrates multiple specialized agents using a supervisor pattern.

**Key Features**:
- Multi-agent coordination
- Specialized agents for different tasks
- Dynamic routing based on query type
- Research, coding, and web search capabilities
- Hierarchical agent architecture

**Architecture Overview**:

The supervisor system implements a hierarchical multi-agent architecture where a supervisor agent routes queries to specialized workers, with a validator agent ensuring quality:

```
                    ┌─────────────┐
                    │   __start__ │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Supervisor  │
                    │   Agent     │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌───────────┐    ┌────────────┐
    │  Coder   │    │ Enhancer  │    │ Researcher │
    │  Agent   │    │  Agent    │    │   Agent    │
    └──────────┘    └───────────┘    └────────────┘
          │                │                 │
          └────────────────┼─────────────────┘
                           ▼
                    ┌─────────────┐
                    │ Validator   │
                    │   Agent     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   __end__   │
                    └─────────────┘
```

**How it Works** (Based on notebook implementation):

1. **Agent Definition**:
   - Each worker agent is defined with specific capabilities and system prompts
   - Agents are registered in the supervisor's routing logic
   - Each agent has access to specialized tools
   - Agents communicate through shared state

2. **Supervisor Logic**:
   ```python
   def supervisor(state: SupervisorState):
       # Analyzes the current state and messages
       # Determines which agent should handle the task
       # Returns routing decision
       return {"next": "coder" | "enhancer" | "researcher" | "validator"}
   ```
   - Uses LLM to intelligently route queries
   - Can route to multiple agents in sequence
   - Tracks conversation state and history

3. **Agent State Management**:
   ```python
   class SupervisorState(TypedDict):
       messages: Annotated[Sequence[BaseMessage], add_messages]
       next: str  # Which agent to route to
   ```

4. **Worker Agent Pattern**:
   Each worker follows this structure:
   ```python
   def worker_agent(state: SupervisorState):
       # Process the task based on agent specialty
       # Use agent-specific tools if needed
       # Return results as messages
       return {"messages": [response]}
   ```

5. **Validator Integration**:
   ```python
   def validator(state: SupervisorState):
       # Reviews outputs from worker agents
       # Checks quality, correctness, completeness
       # Can reject and send back for improvement
       # Final approval before END
       return {"messages": [validation_result]}
   ```

6. **Graph Construction**:
   ```python
   graph = StateGraph(SupervisorState)
   
   # Add all agents as nodes
   graph.add_node("supervisor", supervisor)
   graph.add_node("coder", coder_agent)
   graph.add_node("enhancer", enhancer_agent)
   graph.add_node("researcher", researcher_agent)
   graph.add_node("validator", validator)
   
   # Entry point
   graph.set_entry_point("supervisor")
   
   # Conditional routing from supervisor
   graph.add_conditional_edges(
       "supervisor",
       lambda x: x["next"],
       {
           "coder": "coder",
           "enhancer": "enhancer", 
           "researcher": "researcher",
           "FINISH": "validator"
       }
   )
   
   # All workers route to validator
   graph.add_edge("coder", "validator")
   graph.add_edge("enhancer", "validator")
   graph.add_edge("researcher", "validator")
   
   # Validator can loop back or end
   graph.add_conditional_edges(
       "validator",
       should_continue,
       {
           "continue": "supervisor",
           "end": END
       }
   )
   ```

7. **Workflow Orchestration**:
   - Supervisor receives user query
   - Analyzes query intent and requirements
   - Routes to appropriate specialist(s)
   - Worker(s) complete their tasks
   - Validator ensures quality
   - If approved → Return to user
   - If rejected → Loop back for improvements

3. **Specialized Agents**:

   **Coder Agent**:
   - Writes and debugs code
   - Implements solutions in various programming languages
   - Explains code functionality
   - Provides optimized code solutions

   **Enhancer Agent**:
   - Improves and refines content
   - Optimizes code or text quality
   - Adds missing details and context
   - Polishes outputs from other agents

   **Researcher Agent**:
   - Conducts in-depth research
   - Gathers information from multiple sources
   - Synthesizes comprehensive answers
   - Provides fact-based insights

   **Validator Agent**:
   - Quality assurance for all outputs
   - Verifies correctness and completeness
   - Checks for errors or inconsistencies
   - Ensures outputs meet requirements
   - Final gatekeeper before returning to user

4. **State Management**:
   ```python
   class SupervisorState(TypedDict):
       messages: Annotated[Sequence[BaseMessage], add_messages]
       next: str  # Which agent to route to next
   ```

5. **Routing Logic**:
   - Supervisor examines query
   - Uses LLM to determine appropriate agent
   - Can chain multiple agents for complex queries

6. **Workflow**:
   ```
   User Query → Supervisor → [Route Decision] → Worker Agent(s) 
                                                      ↓
                                                  Validator
                                                      ↓
                                                   Result
   ```

   **Complete Flow**:
   1. User submits query
   2. Supervisor analyzes and routes to appropriate agent(s)
   3. Worker agent(s) process the task
   4. Validator checks the output quality
   5. If valid → Return to user
   6. If invalid → Route back to worker or supervisor for improvement

**Key Notebook Components**:
- Agent initialization and configuration
- Supervisor prompt engineering
- Graph construction with conditional edges
- Multi-agent communication protocol
- Result aggregation and synthesis

**Detailed Agent Capabilities**:

| Agent | Primary Role | Key Functions | Example Tasks |
|-------|-------------|---------------|---------------|
| **Supervisor** | Orchestrator | Query analysis, routing, coordination | Determines workflow path |
| **Coder** | Software Development | Write code, debug, optimize, explain | "Create a Python sorting algorithm" |
| **Enhancer** | Quality Improvement | Refine outputs, add details, polish | "Improve this code's readability" |
| **Researcher** | Information Gathering | Research topics, synthesize data | "Find latest ML trends" |
| **Validator** | Quality Assurance | Check correctness, verify completeness | Ensures all outputs meet standards |

**Agent Interaction Patterns**:

1. **Sequential Processing**:
   ```
   Researcher → Coder → Enhancer → Validator
   ```
   Example: "Research topic, write code, optimize it"

2. **Single Agent + Validation**:
   ```
   Coder → Validator
   ```
   Example: "Write a simple function"

3. **Iterative Improvement**:
   ```
   Coder → Validator → (fails) → Enhancer → Validator
   ```
   Example: Quality loop until standards met

4. **Parallel-to-Sequential** (if supported):
   ```
   Researcher + Coder → Enhancer → Validator
   ```
   Example: Combine research and code, then polish

**Use Cases**:
- Complex queries requiring multiple specialized skills
- Research + coding + optimization workflows
- Quality-assured content generation
- Multi-step problem solving with validation
- Comprehensive analysis requiring diverse agents
- Production-ready outputs with automatic QA

**Example Flow**:
```
User: "Research the latest AI trends and write optimized Python code to analyze them"

Supervisor: Analyzes query → "This needs Researcher, Coder, and Enhancer"
↓
Researcher: Finds information about AI trends
↓
Supervisor: Receives research → Routes to Coder
↓
Coder: Writes analysis code based on research
↓
Supervisor: Routes to Enhancer
↓
Enhancer: Optimizes code, adds documentation and best practices
↓
Validator: Checks if research is complete, code works, and quality is high
↓
Validator: Approves → Returns final result to user

Alternative Flow (if validation fails):
Validator: Finds issues → Routes back to Supervisor
↓
Supervisor: Re-routes to appropriate agent for fixes
↓
[Process repeats until validation passes]
```

**Advanced Features**:
- Agent memory and context sharing
- Parallel agent execution (if supported)
- Error handling and fallback strategies
- Agent communication protocols

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key
- **UV package manager** (highly recommended - fast, modern Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Abdul-Halim01/LangGraph-MultiAgents.git
cd LangGraph-MultiAgents
```

### Step 2: Install UV (if not already installed)

**On macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or using pip:**
```bash
pip install uv
```

**Verify installation:**
```bash
uv --version
```

### Step 3: Set Up Virtual Environment with UV

```bash
# UV automatically creates and manages virtual environments
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 4: Install Dependencies with UV

UV makes dependency installation fast and reliable:

```bash
# Install all dependencies from pyproject.toml
uv pip install -e .

# Or install specific packages
uv pip install langgraph langchain langchain-core langchain-google-genai
uv pip install langchain-community langchain-chroma langchain-huggingface
uv pip install python-dotenv pandas jupyter

# UV also supports syncing dependencies
uv pip sync
```

**Why UV?**
- ⚡ **10-100x faster** than pip
- 🔒 **Reliable**: Generates lock files for reproducible installs
- 🎯 **Smart**: Better dependency resolution
- 💾 **Efficient**: Global cache reduces disk usage

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Get your Gemini API key**:
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste into `.env`

---

## 🚀 UV Package Manager

This project uses **UV** as the primary package manager for superior performance and reliability.

### What is UV?

UV is a modern, extremely fast Python package installer and resolver written in Rust. It's designed to be a drop-in replacement for pip with significantly better performance.

### Benefits of UV

| Feature | UV | pip |
|---------|-----|-----|
| **Speed** | ⚡ 10-100x faster | Standard speed |
| **Dependency Resolution** | 🎯 Advanced resolver | Basic resolver |
| **Lock Files** | ✅ Built-in (uv.lock) | ❌ Requires pip-tools |
| **Caching** | 💾 Global cache | Limited caching |
| **Reproducibility** | 🔒 Guaranteed | Variable |

### UV Commands Quick Reference

```bash
# Create virtual environment
uv venv

# Install from pyproject.toml
uv pip install -e .

# Install specific package
uv pip install package-name

# Install with extras
uv pip install "package-name[extra]"

# Sync dependencies (use lock file)
uv pip sync

# List installed packages
uv pip list

# Freeze dependencies
uv pip freeze

# Uninstall package
uv pip uninstall package-name
```

### UV Lock File

The `uv.lock` file ensures everyone on your team has the exact same dependencies:

```bash
# Generate/update lock file
uv pip compile pyproject.toml -o requirements.txt

# Install from lock file
uv pip sync
```

### Migration from pip

If you're coming from pip, UV commands are nearly identical:

```bash
# pip → uv
pip install package    →  uv pip install package
pip uninstall package  →  uv pip uninstall package
pip list              →  uv pip list
pip freeze            →  uv pip freeze
```

---

## ⚙️ Configuration

### LangGraph Configuration (`langgraph.json`)

```json
{
  "dependencies": ["langgraph"],
  "graphs": {
    "agent": "./Drafter.py:app"
  }
}
```

This configuration file:
- Specifies LangGraph as a dependency
- Defines the Drafter agent as the default graph for LangGraph API deployment

### Python Version

Specified in `.python-version`:
```
3.12
```

### Project Dependencies (`pyproject.toml`)

Key dependencies:
- `langgraph`: Graph-based agent orchestration
- `langchain`: Core LLM framework
- `langchain-google-genai`: Gemini AI integration
- `langchain-community`: Community tools and integrations
- `langchain-chroma`: Vector store integration
- `langchain-huggingface`: HuggingFace embeddings
- `python-dotenv`: Environment variable management
- `pandas`: Data manipulation

---

## 🚀 Usage

### Running Individual Agents

#### 1. Data Analysis Agent
```bash
python agent.py
```
**Example queries**:
- "What are the average sales?"
- "Show me the total profit"
- "What's the sales trend over time?"

#### 2. Drafter Agent
```bash
python Drafter.py
```
**Interactive session**:
```
USER: Create a product description for a new smartphone
AI: [Updates document with description]
USER: Make it more technical
AI: [Updates with technical details]
USER: Save as product_desc
AI: [Saves to product_desc.txt]
```

#### 3. Lab Agent (Conversational)
```bash
python Lab_agent.py
```
**Example**:
```
Enter your message: Hello!
AI: Hi! How can I help you today?
Enter your message: Tell me a joke
AI: [Tells a joke]
Enter your message: exit
Conversation Saved to logging.txt!
```

#### 4. RAG Agent
```bash
python RAG_Agent.py
```
**Ensure PDF exists**: `Stock_Market_Performance_2024.pdf`

**Example queries**:
- "What were the key market trends in 2024?"
- "Which sectors performed best?"
- "What caused the Q2 market volatility?"

#### 5. ReAct Agent
```bash
python ReAct_Agent.py
```
**Example**:
```python
# Already has an example in the code
inputs = {"messages": [("user", "Add 34 + 21 + 7")]}
# Outputs the reasoning and calculation result
```

#### 6. Supervisor Multi-Agent System

**Installation:**
```bash
# Install Jupyter with UV
uv pip install jupyter ipykernel

# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=langgraph-multiagents
```

**Run:**
```bash
# Launch Jupyter
jupyter notebook Supervisor.ipynb

# Or use Jupyter Lab
uv pip install jupyterlab
jupyter lab Supervisor.ipynb
```

Then run all cells to initialize and interact with the multi-agent system.

**Architecture:**
- **Supervisor**: Routes tasks to specialized agents
- **Coder**: Handles programming tasks
- **Enhancer**: Improves and optimizes outputs
- **Researcher**: Gathers and synthesizes information
- **Validator**: Ensures output quality and correctness

---

## 🔧 Technical Details

### State Management

All agents use TypedDict for type-safe state management:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**`add_messages` reducer**: Automatically appends new messages to the message list.

### Graph Construction Pattern

Standard pattern across agents:

```python
from langgraph.graph import StateGraph, START, END

# 1. Create graph with state type
graph = StateGraph(AgentState)

# 2. Add nodes
graph.add_node("agent", agent_function)
graph.add_node("tools", tool_node)

# 3. Set entry point
graph.set_entry_point("agent")

# 4. Add edges
graph.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
graph.add_edge("tools", "agent")

# 5. Compile
app = graph.compile()
```

### Tool Creation

Using `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description for LLM"""
    # Tool logic
    return result
```

**Important**:
- Docstring is used by LLM to understand tool purpose
- Type hints are required
- Return type should be string for consistency

### Conditional Routing

Pattern for decision-making in graphs:

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    
    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"  # Route to tools
    return "end"  # Terminate

# Use in graph
graph.add_conditional_edges(
    "node_name",
    should_continue,
    {"continue": "next_node", "end": END}
)
```

### Message Types

LangChain message types used:

- **`HumanMessage`**: User input
- **`AIMessage`**: LLM response
- **`SystemMessage`**: System prompts and instructions
- **`ToolMessage`**: Tool execution results

### Gemini Model Configuration

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",      # Model version
    temperature=0.1,                # Creativity (0-1)
    max_tokens=500,                 # Response length limit
    max_retries=2,                  # Retry failed requests
)
```

**Model Options**:
- `gemini-2.5-flash`: Fast, efficient, good for most tasks
- `gemini-2.5-flash-lite`: Lightweight version
- `gemini-2.5-pro`: More capable, slower, higher cost

### RAG Components

**Document Processing**:
```python
# Load
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap to maintain context
)
docs = splitter.split_documents(pages)

# Embed and Store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# Retrieve
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Top 5 results
)
```

### Error Handling

Best practices implemented:

```python
try:
    # Tool execution
    result = tool.invoke(input)
except Exception as e:
    return f"Error: {str(e)}"
```

All tools include try-except blocks for graceful error handling.

---

## 📦 Dependencies

### Core Dependencies

```toml
[project]
dependencies = [
    "langgraph>=0.2.59",
    "langchain>=0.3.14",
    "langchain-core>=0.3.28",
    "langchain-google-genai>=2.0.8",
    "langchain-community>=0.3.14",
    "langchain-chroma>=0.2.0",
    "langchain-huggingface>=0.1.2",
    "python-dotenv>=1.0.0",
    "pandas>=2.0.0",
]
```

### Optional Dependencies

For enhanced functionality:
- `jupyter`: For running Supervisor notebook
- `pypdf`: PDF processing (included in langchain-community)
- `chromadb`: Vector database
- `sentence-transformers`: Embeddings

### Installation with UV

```bash
# Quick install with UV (recommended)
uv pip install langgraph langchain langchain-google-genai python-dotenv

# Full installation with all features
uv pip install -e .
```

### Alternative Installation (pip)

If you prefer traditional pip:

```bash
pip install langgraph langchain langchain-google-genai python-dotenv
pip install langchain-community langchain-chroma langchain-huggingface pandas
```

**Note**: UV is significantly faster and more reliable for dependency management.

---

## 🎓 Key Concepts Explained

### LangGraph
A library for building stateful, multi-actor applications with LLMs. Key features:
- **State management**: TypedDict-based state
- **Graph construction**: Nodes and edges
- **Conditional routing**: Dynamic workflow control
- **Tool integration**: Seamless tool calling

### ReAct Pattern
**Re**asoning + **Act**ing:
1. LLM reasons about the problem
2. LLM decides to use a tool (acting)
3. Tool executes and returns observation
4. LLM reasons with new information
5. Repeat or provide final answer

### RAG (Retrieval-Augmented Generation)
Enhances LLM with external knowledge:
1. User asks a question
2. System retrieves relevant documents
3. LLM generates answer using retrieved context
4. Reduces hallucinations, provides citations

### Supervisor Pattern
Multi-agent orchestration:
- Central supervisor coordinates worker agents
- Each agent specializes in specific tasks
- Supervisor routes queries to appropriate agents
- Enables complex, multi-step workflows

---

## 🛠️ Customization Guide

### Adding New Tools

```python
@tool
def your_custom_tool(input: str) -> str:
    """Description for the LLM"""
    # Your logic here
    return result

# Add to tools list
tools = [existing_tools, your_custom_tool]
llm_with_tools = llm.bind_tools(tools)
```

### Modifying System Prompts

Each agent has a system prompt you can customize:

```python
system_prompt = """
Your custom instructions here.
- Guideline 1
- Guideline 2
"""
```

### Changing LLM Models

```python
# Switch to different Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # More capable
    temperature=0.7,          # More creative
    max_tokens=1000,          # Longer responses
)
```

### Adding New Agents

1. Create a new Python file
2. Define state and tools
3. Create graph with nodes and edges
4. Compile and test
5. Integrate with supervisor (optional)

---

## 📊 Performance Considerations

### Model Selection
- **Flash models**: Fast, cost-effective for most tasks
- **Pro models**: Better for complex reasoning
- **Lite models**: Minimal latency, good for simple tasks

### Chunking Strategy
For RAG agents:
- **Chunk size**: 500-1500 characters (balance context vs. precision)
- **Overlap**: 10-20% of chunk size
- **Smaller chunks**: More precise retrieval, less context
- **Larger chunks**: More context, potentially less precise

### Vector Store
- **ChromaDB**: Good for local development, persists to disk
- **Pinecone**: Better for production, managed service
- **FAISS**: Fast, in-memory, no persistence

### Caching
- Enable LLM caching for repeated queries
- Cache embeddings for frequently accessed documents
- Use persistent vector stores to avoid re-indexing

---

## 🐛 Troubleshooting

### Common Issues

**1. UV Not Found**
```bash
-bash: uv: command not found
```
**Solution**: Install UV
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**2. UV Installation Fails**
```bash
error: externally-managed-environment
```
**Solution**: Use virtual environment
```bash
uv venv
source .venv/bin/activate  # Activate first
uv pip install package-name
```

**3. Import Errors**
```bash
ModuleNotFoundError: No module named 'langgraph'
```
**Solution**: Install dependencies with UV
```bash
uv pip install langgraph langchain
# Or install everything
uv pip install -e .
```

**4. Virtual Environment Not Activated**
```bash
# Check if venv is activated (should see (.venv) in prompt)
which python  # Should point to .venv/bin/python

# If not activated:
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

**7. ChromaDB Permission Issues**
```bash
PermissionError: Cannot write to ./chroma_db
```
**Solution**: Check directory permissions
```bash
chmod 755 ./chroma_db
# Or change persist_directory in RAG_Agent.py
```


**8. Memory Issues with Large PDFs**
**Solution**: 
```python
# Reduce chunk_size in RAG_Agent.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Reduced from 1000
    chunk_overlap=100,
)
```

**9. UV Cache Issues**
```bash
# Clear UV cache if experiencing issues
uv cache clean
```

---

## 🌟 Best Practices

1. **Environment Variables**: Always use `.env` for API keys
2. **Error Handling**: Wrap tool execution in try-except blocks
3. **Type Hints**: Use TypedDict for state management
4. **Logging**: Implement logging for production deployments
5. **Testing**: Test each agent independently before integration
6. **Documentation**: Keep docstrings updated for tools
7. **Version Control**: Don't commit `.env` files
8. **Resource Cleanup**: Close connections and clean up resources

---
## 📚 Resources

### LangGraph Documentation
- [Official Docs](https://langchain-ai.github.io/langgraph/)
- [Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/)

### Gemini AI
- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Pricing](https://ai.google.dev/pricing)

### LangChain
- [Documentation](https://python.langchain.com/)
- [Community](https://github.com/langchain-ai/langchain)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow existing code style
- Add tests for new features
- Update documentation
- Ensure all agents still work

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👨‍💻 Author

**Abdul-Halim01**
- GitHub: [@Abdul-Halim01](https://github.com/Abdul-Halim01)

---

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- Google for Gemini AI
- LangGraph community for inspiration
- All contributors and users

<!-- ---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing issues and discussions
- Review documentation and examples

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/Abdul-Halim01/LangGraph-MultiAgents)
![GitHub forks](https://img.shields.io/github/forks/Abdul-Halim01/LangGraph-MultiAgents)
![GitHub issues](https://img.shields.io/github/issues/Abdul-Halim01/LangGraph-MultiAgents)

--- -->

**Happy Building with LangGraph! 🚀**

---

<!-- *Last Updated: February 2026* -->