"""
Retrieval agent implementation for the Agentic RAG Medical Documentation System.
Creates and manages the multi-tool agent for information retrieval.
"""

from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate

from config.llm_config import llm_config
from config.settings import MAX_AGENT_ITERATIONS, AGENT_EARLY_STOPPING_METHOD, CHAT_HISTORY_LIMIT
from core.memory import memory_manager
from tools.patient_tools import retrieve_patient_info
from tools.guideline_tools import retrieve_guideline_knowledge
from tools.document_tools import retrieve_generated_document_info, start_document_generation
from utils.profiling import profile

def create_retrieval_agent() -> AgentExecutor:
    """
    Creates a retrieval agent with appropriate tools, memory, and configuration.
    The agent uses a conversational memory buffer and multiple specialized tools
    to retrieve and process information from guidelines, patient records, and
    generated documents.
    
    Returns:
        AgentExecutor: An initialized agent executor ready to process queries.
    """
    # Load memory content
    memory_variables = memory_manager.retrieval_memory.load_memory_variables({})
    chat_history = memory_variables.get("chat_history", [])
    
    # Limit chat history to the last conversations
    chat_history = chat_history[-CHAT_HISTORY_LIMIT:]
    
    # Format chat history for inclusion in the prompt
    formatted_history = ""
    if chat_history:
        for message in chat_history:
            if hasattr(message, "content") and hasattr(message, "type"):
                role = "Human" if message.type == "human" else "Assistant"
                formatted_history += f"{role}: {message.content}\n"
    
    # Initialize tools for the agent
    tools = [retrieve_guideline_knowledge, retrieve_patient_info, retrieve_generated_document_info, start_document_generation]
    
    # Create offline-compatible prompt (no hub dependency)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an intelligent multi-step retrieval (RAG) agent working in a hospital. 
        You are expected to think aloud, reason step-by-step, and use multiple tools if needed to fully understand and answer the user's question. 
        
        Recent conversation history:
        {formatted_history}
    
        Please refer to this conversation history when answering questions that reference previous interactions.
        Your goal is to simulate how an expert clinician-researcher would retrieve and synthesize information from multiple sources to give a clear, accurate answer – in Danish.

        Rules:
            - Use the 'start_document_generation' tool if the user asks to create a document
            - Never repeat the same tool with the same input more than once.
            - Stop when you have enough information.
            - Always answer in Danish.
        
        Available tools:
            - retrieve_patient_info: Access patient records, input your search criterias for the vectordatabase in danish
            - retrieve_generated_document_info: Get information about finalized documents, input your search criterias for the vectordatabase in danish
            - start_document_generation: Launch creation of a medical document. Use this if the user asks you to create a document.
            - retrieve_guideline_knowledge: Get information about medical guidelines, input your search criterias for the vectordatabase in danish
        
        Think step-by-step:
            - First, analyze the user's question in detail.
            - Then hypothesize what information you may need to answer it.
            - Then decide which tool is best for retrieving that information.
            - After each tool, reflect: did this bring you closer to the answer? Should you expand, follow up, or rerank?
            - You are allowed to perform multiple tool actions in sequence before finalizing your answer, especially when the question requires combining patient-specific and guideline-based knowledge.
            - Before you finalize, make sure to check if the information you have is sufficient to answer the question. If any information is redundant or irrelevant, you should remove it.
        
        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {{tool_names}}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{{{
         "action": $TOOL_NAME,
         "action_input": $INPUT
        }}}}
        ```
        Where $INPUT is your query for the RAG.

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{{{
         "action": "Final Answer",
         "action_input": "Final response to human"
        }}}}
        ```"""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])
    
    prompt = prompt.partial(tools=", ".join([tool.name for tool in tools]))

    agent = create_structured_chat_agent(
        llm=llm_config.llm_retrieve,
        tools=tools,
        prompt=prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory_manager.retrieval_memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=MAX_AGENT_ITERATIONS,
        early_stopping_method=AGENT_EARLY_STOPPING_METHOD
    )
    
    return agent_executor

@profile
def invoke_retrieval_agent(query: str):
    """
    Uses an LLM agent to collect information for answering the user query as well as generating sub-sections of the document.
    This function is profiled for performance monitoring.
    
    Args:
        query (str): The user's query to be processed by the agent.
        
    Returns:
        str: The agent's response to the query.
    """
    agent = create_retrieval_agent()
    
    try:
        print(f"[DEBUG] Invoking retrieval agent with query: {query}")
        response = agent.invoke(
            input={"input": query}
        )

        # Check if the response is a dictionary or string
        if isinstance(response, dict) and 'output' in response:
            return response['output']
        elif isinstance(response, str):
            return response
        else:
            return "Unexpected agent response format."
    except Exception as e:
        return f"Retrieval agent failed: {str(e)}"