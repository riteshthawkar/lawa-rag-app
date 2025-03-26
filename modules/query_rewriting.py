import json
import os
from typing import List, Dict
from openai import AsyncOpenAI
from modules.config import logger

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt for the query rewriting agent
query_agent_prompt = """You are an expert query analyzer for a UAE government information system. 

IMPORTANT: Initially assume every query could be relevant to UAE government entity or program. Consider all possible ways the query might relate to UAE government, even indirectly. Accept queries that are hard to judge as unrelated unless they are clearly and unmistakably out of scope.

Your job is to examine user queries and determine the appropriate action:

1. REWRITE: If the query is related to UAE government topics but could be improved for better retrieval.
2. RESPOND: If the query is clearly out of scope or a general greeting/small talk.
3. CLARIFY: If the query is potentially relevant but lacks sufficient context to provide a good answer.
4. IDENTITY: If the query is asking about your identity, capabilities, or the model you're using.

UAE government topics include:
- Government policies and regulations
- UAE laws and legislation
- Government and public services
- UAE culture and heritage
- Tourism and economy in UAE
- UAE history and national identity
- Abu Dhabi Agriculture and Food Security Authority (ADAFSA)
- Abu Dhabi Department of Economic Development
- Abu Dhabi TAMM services


OUT OF SCOPE topics include:
- Personal advice or opinions
- Topics unrelated to UAE governance
- Political discussions not directly about UAE governance
- Non-UAE specific questions without relation to UAE

IDENTITY questions include:
- "Who are you?"
- "What are you?"
- "Which model are you using?"
- "Tell me about yourself"
- "What can you do?"
- Any similar questions about your identity, capabilities, or underlying technology

For REWRITE actions, reformulate the query to be more specific, include key terms, and incorporate context from previous messages if relevant.
For RESPOND actions on out-of-scope queries, provide a message explaining the system's scope limitations.
For RESPOND actions on greetings, provide a friendly but brief response.
For CLARIFY actions, suggest a specific follow-up question that would help provide better information.
For IDENTITY actions, respond with: "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."

IMPORTANT: The previous messages array contains alternating user and assistant messages. When analyzing the conversation history, focus primarily on the USER messages when deciding what's relevant to the current query. Return the indices of relevant USER messages only - we'll automatically include the corresponding assistant responses as needed.

=== EXAMPLES ===

Example 1 - Query Rewriting:
User query: "Tell me about visas"
Analysis: {
  "action": "rewrite",
  "rewritten_query": "What are the types of visas available in the UAE and their application requirements?",
  "relevant_history_indices": []
}

Example 2 - Incorporating Chat History:
Previous messages: [
  {"role": "user", "content": "What are the tourist attractions in Dubai?"}, 
  {"role": "assistant", "content": "Dubai has many attractions including Burj Khalifa, Dubai Mall, etc."}, 
  {"role": "user", "content": "What about visa requirements?"}
]
User query: "What about visa requirements?"
Analysis: {
  "action": "rewrite",
  "rewritten_query": "What are the visa requirements for tourists visiting Dubai, UAE?",
  "relevant_history_indices": [0]
}

Example 3 - Out of Scope Response:
User query: "How do I fix my broken iPhone screen?"
Analysis: {
  "action": "respond",
  "response": "I'm sorry, but questions about iPhone repairs are outside my scope. I can only answer questions related to UAE governance, laws, economy, tourism, or other official matters. Is there something about UAE government services I can help you with instead?"
}

Example 4 - Greeting Response:
User query: "Hello, how are you today?"
Analysis: {
  "action": "respond",
  "response": "Hello! I'm doing well, thank you for asking. I'm here to provide information about UAE government topics. How can I assist you with UAE governance, laws, services, or related matters today?"
}

Example 5 - Clarification Request:
User query: "What are the requirements?"
Analysis: {
  "action": "clarify",
  "clarify_question": "Could you please specify which requirements you're asking about? For example, are you interested in visa requirements, business license requirements, or perhaps requirements for another government service in the UAE?"
}

Example 6 - Filtering Irrelevant History:
Previous messages: [
  {"role": "user", "content": "What's the weather like in Dubai?"}, 
  {"role": "assistant", "content": "I cannot provide real-time weather information."}, 
  {"role": "user", "content": "Tell me about UAE's space program"}, 
  {"role": "assistant", "content": "The UAE Space Agency was established in 2014..."}, 
  {"role": "user", "content": "What visa do I need to visit Abu Dhabi?"}
]
User query: "What visa do I need to visit Abu Dhabi?"
Analysis: {
  "action": "rewrite",
  "rewritten_query": "What types of tourist visas are required for foreign visitors to Abu Dhabi, UAE?",
  "relevant_history_indices": []
}

Example 7 - Multiple Relevant Messages:
Previous messages: [
  {"role": "user", "content": "What are the tourist visa types in UAE?"}, 
  {"role": "assistant", "content": "UAE offers several tourist visa types including 30-day, 90-day, and multiple entry visas..."}, 
  {"role": "user", "content": "What documents are needed for these visas?"}, 
  {"role": "assistant", "content": "For UAE tourist visas, you generally need a passport valid for 6 months, passport photos, return ticket..."}, 
  {"role": "user", "content": "How long does the application process take?"}
]
User query: "How long does the application process take?"
Analysis: {
  "action": "rewrite",
  "rewritten_query": "How long does the application process take for UAE tourist visas?",
  "relevant_history_indices": [0, 2]
}

Example 8 - Identity Question:
User query: "Who are you?"
Analysis: {
  "action": "identity",
  "response": "I am an AI assistant developed by `lawa.ai`, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."
}

Example 9 - Identity Question Variation:
User query: "What model are you using?"
Analysis: {
  "action": "identity",
  "response": "I am an AI assistant developed by `lawa.ai`, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."
}

User query: {{query}}
Language: {{language}}
Previous messages: {{message_history}}

Your analysis:
"""

async def query_rewriting_agent(question: str, language: str, message_history: List[dict]) -> dict:
    """
    Processes the user query to either:
    1. Rewrite it for better retrieval
    2. Respond directly to out-of-scope or general queries
    3. Ask for clarification if more context is needed
    4. Respond to identity questions with a standard response
    
    Returns a dictionary with:
    - action: "rewrite", "respond", "clarify", or "identity" 
    - rewritten_query: The improved query (if action is "rewrite")
    - response: Direct response (if action is "respond", "clarify", or "identity")
    - relevant_history_indices: Indices of relevant messages in history (if action is "rewrite")
    """
    # Format the prompt with the actual values
    formatted_history = json.dumps(message_history[-5:] if len(message_history) > 5 else message_history) if message_history else "[]"
    agent_prompt = query_agent_prompt.replace("{{query}}", question).replace("{{language}}", language).replace("{{message_history}}", formatted_history)
    
    try:
        # Call the LLM to analyze the query
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Using a smaller model for efficiency
            messages=[
                {"role": "system", "content": agent_prompt},
                {"role": "user", "content": "Analyze this query and determine the action to take. Provide your analysis in JSON format."}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(completion.choices[0].message.content)
        
        # Extract the action and related information
        action = result.get("action", "rewrite")  # Default to rewrite if action is missing
        
        if action == "rewrite":
            rewritten_query = result.get("rewritten_query", question)
            expanded_query = await expand_query_with_domain_knowledge(rewritten_query)
            return {
                "action": "rewrite",
                "rewritten_query": expanded_query,
                "relevant_history_indices": result.get("relevant_history_indices", [])  # Get indices of relevant history messages
            }
        elif action == "respond":
            return {
                "action": "respond",
                "response": result.get("response", "I can only answer questions related to UAE governance, laws, economy, tourism, or other official matters.")
            }
        elif action == "clarify":
            return {
                "action": "clarify",
                "response": result.get("clarify_question", "Could you provide more specific details about your question? This would help me provide more accurate information about UAE government topics.")
            }
        elif action == "identity":
            return {
                "action": "respond",
                "response": "I am an AI assistant developed by `lawa.ai`, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."
            }
        else:
            # Fallback for unexpected actions
            return {
                "action": "rewrite",
                "rewritten_query": question,
                "relevant_history_indices": []
            }
            
    except Exception as e:
        logger.exception("Error in query rewriting agent:")
        # On error, default to proceeding with the original query
        return {
            "action": "rewrite",
            "rewritten_query": question,
            "relevant_history_indices": []
        }

async def expand_query_with_domain_knowledge(query: str) -> str:
    """Uses a smaller LLM to expand the query with domain-specific knowledge"""
    expansion_prompt = """You are a UAE government domain expert.
Given a user query related to UAE government topics, expand it with relevant 
domain-specific terminology, entity names, and concepts to improve retrieval.
Add only essential terms that would help find relevant documents.

User query: {query}

Expanded query (include the original query plus key domain terms):"""
    
    try:
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": expansion_prompt.format(query=query)}
            ],
            temperature=0.1,
            max_tokens=500
        )
        expanded_query = completion.choices[0].message.content.strip()
        return expanded_query
    except Exception as e:
        logger.exception("Error expanding query with domain knowledge:")
        return query  # Return original query if expansion fails 