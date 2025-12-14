import json
import os
from typing import List, Dict
from openai import AsyncOpenAI
from modules.config import logger, QUERY_REWRITING_TEMPERATURE, QUERY_REWRITING_MODEL, QUERY_TYPES

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt for the query rewriting agent
query_agent_prompt = """You are an expert query analyzer for a UAE government information system. 

IMPORTANT: Initially assume every query could be relevant to UAE government entity or program. Consider all possible ways the query might relate to UAE government, even indirectly. Accept queries that are hard to judge as unrelated unless they are clearly and unmistakably out of scope.

IMPORTANT: Take a BROAD PERSPECTIVE when judging relevance. Many seemingly personal topics may relate to government services:
- Questions about animal health, agriculture, or food relate to Abu Dhabi Agriculture and Food Security Authority (ADAFSA)
- Questions about business, employment, or trade relate to Economic Development departments
- Questions about housing, property, or visas relate to various government services

IMPORTANT: Be smart about clarification requests. Only ask for clarification when the query is genuinely ambiguous or too vague to provide a useful answer. Consider these scenarios:
- Very broad queries like "visas" or "requirements" without context
- Queries that could refer to multiple government services
- Queries that lack specific details needed for a complete answer

Your job is to examine user queries and determine the appropriate action:

1. REWRITE: If the query is related to UAE government topics and can be improved for better retrieval.
2. RESPOND: If the query is clearly out of scope, a general greeting/small talk, or if you can provide a direct answer.
3. CLARIFY: If the query is potentially relevant but too vague or ambiguous to provide a good answer.
4. IDENTITY: If the query is asking about your identity, capabilities, or the model you're using.
Also classify the query into one of the following types:
{{query_types_prompt}}

If the query does not match any of the above types, classify it as 'Other' and specify the type, e.g., 'Other: greeting' or 'Other: nonsense'.

UAE government topics include:
- Government policies and regulations
- UAE laws and legislation
- Government and public services
- UAE culture and heritage
- Tourism and economy in UAE
- UAE history and national identity
- Abu Dhabi Agriculture and Food Security Authority (ADAFSA) including animal health, agriculture, and food safety
- Abu Dhabi Department of Economic Development
- Abu Dhabi TAMM services


OUT OF SCOPE topics include:
- Personal advice or opinions unrelated to government services
- Topics completely unrelated to UAE governance or public services
- Political discussions not directly about UAE governance
- Non-UAE specific questions without relation to UAE

IDENTITY questions include:
- "Who are you?"
- "What are you?"
- "Which model are you using?"
- "Tell me about yourself"
- "What can you do?"
- Any similar questions about your identity, capabilities, or underlying technology

For REWRITE actions, reformulate the query to be more specific, include key terms, and incorporate context from previous messages if relevant. CRITICAL: If the input query is in a non-English language (e.g., Arabic), the `rewritten_query` MUST be translated into ENGLISH to ensure effective retrieval from the English knowledge base.
For RESPOND actions on out-of-scope queries, provide a message explaining the system's scope limitations (in the user's language).
For RESPOND actions on greetings, provide a friendly but brief response.
For CLARIFY actions, suggest a specific follow-up question that would help provide better information.
For IDENTITY actions, respond with: "I am an AI assistant developed by Lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."

IMPORTANT: The previous messages array contains alternating user and assistant messages. When analyzing the conversation history, focus primarily on the USER messages when deciding what's relevant to the current query. Return the indices of relevant USER messages only - we'll automatically include the corresponding assistant responses as needed.

=== EXAMPLES ===

Example 1 - Query Rewriting:
User query: "Tell me about visas"
Analysis: {
  "action": "rewrite",
  "rewritten_query": "What are the types of visas available in the UAE and their application requirements?",
  "query_type": "General Information",
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
  "response": "I'd be happy to help you with requirements information! Could you please specify which type of requirements you're interested in? For example:\n- Visa requirements for visiting UAE\n- Business license requirements\n- Employment requirements\n- Property ownership requirements\n- Or any other specific government service requirements?"
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
  "response": "I am an AI assistant developed by Lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."
}

Example 9 - Identity Question Variation:
User query: "What model are you using?"
Analysis: {
  "action": "identity",
  "response": "I am an AI assistant developed by Lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."
}

Example 10 - Animal Health Query:
User query: "My camel is sick, how can I get help?"
Analysis: {
  "action": "rewrite",
  "rewritten_query": "What veterinary services or support does ADAFSA (Abu Dhabi Agriculture and Food Security Authority) provide for sick camels in UAE?",
  "relevant_history_indices": []
}

Example 11 - Ambiguous Query Clarification:
User query: "visas"
Analysis: {
  "action": "clarify",
  "response": "I can help you with visa information! Could you please specify what type of visa you're interested in?\n\n- Tourist visas for visiting UAE\n- Work visas for employment\n- Student visas for education\n- Business visas for commercial activities\n- Transit visas for short stays\n- Or any other specific visa type?"
}

Example 12 - Vague Service Query:
User query: "How do I apply?"
Analysis: {
  "action": "clarify",
  "response": "I'd be happy to help you with the application process! Could you please specify what you'd like to apply for?\n\n- UAE visa application\n- Business license application\n- Government service application\n- Employment application\n- Property registration\n- Or any other specific application process?"
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
    - query_type: The classification of the query
    """
    # Format the prompt with the actual values
    formatted_history = json.dumps(message_history[-5:] if len(message_history) > 5 else message_history) if message_history else "[]"
    
    # Dynamically generate the query types section
    query_types_str = "\n".join([f"- '{key}': {value}" for key, value in QUERY_TYPES.items()])
    
    agent_prompt = query_agent_prompt.replace("{{query}}", question).replace("{{language}}", language).replace("{{message_history}}", formatted_history).replace("{{query_types_prompt}}", query_types_str)
    
    try:
        # Call the LLM to analyze the query
        completion = await openai_client.chat.completions.create(
            model=QUERY_REWRITING_MODEL,
            messages=[
                {"role": "system", "content": agent_prompt},
                {"role": "user", "content": "Analyze this query and determine the action to take. Provide your analysis in JSON format."}
            ],
            temperature=QUERY_REWRITING_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(completion.choices[0].message.content)
        
        # Extract the action and related information
        action = result.get("action", "rewrite")  # Default to rewrite if action is missing
        query_type = result.get("query_type", "General Information")
        
        if action == "rewrite":
            rewritten_query = result.get("rewritten_query", question)
            expanded_queries = await expand_query_with_domain_knowledge(rewritten_query)
            return {
                "action": "rewrite",
                "rewritten_queries": expanded_queries,
                "rewritten_query": expanded_queries[0] if expanded_queries else rewritten_query, # Fallback/Primary
                "relevant_history_indices": result.get("relevant_history_indices", []),  # Get indices of relevant history messages
                "query_type": query_type
            }
        elif action == "respond":
            return {
                "action": "respond",
                "response": result.get("response", "I can only answer questions related to UAE governance, laws, economy, tourism, or other official matters."),
                "query_type": query_type
            }
        elif action == "clarify":
            return {
                "action": "clarify",
                "response": result.get("response", "Could you provide more specific details about your question? This would help me provide more accurate information about UAE government topics."),
                "query_type": "Clarification"
            }
        elif action == "identity":
            return {
                "action": "respond",
                "response": "I am an AI assistant developed by Lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics.",
                "query_type": "Identity"
            }
        else:
            # Fallback for unexpected actions
            return {
                "action": "rewrite",
                "rewritten_queries": [question],
                "rewritten_query": question,
                "relevant_history_indices": [],
                "query_type": "Other"
            }
            
    except Exception as e:
        logger.exception("Error in query rewriting agent:")
        # On error, default to proceeding with the original query
        return {
            "action": "rewrite",
            "rewritten_queries": [question],
            "rewritten_query": question,
            "relevant_history_indices": [],
            "query_type": "Error"
        }

async def expand_query_with_domain_knowledge(query: str) -> List[str]:
    """Uses a smaller LLM to expand the query into multiple diverse search queries"""
    expansion_prompt = """You are a UAE government domain expert and search optimization specialist.
Given a user query, generate 3 distinctly different search queries in ENGLISH to improve document retrieval (even if the input is in Arabic or another language).

Strategies to use:
1. Domain Expansion: Add specific UAE government entity names and terminology.
2. Semantic Variation: Use synonyms and related concepts.
3. Question Rewording: Rephrase the question from different angles.

User query: {query}

Output valid JSON ONLY in this format:
{{
    "queries": [
        "expanded query 1 with domain terms",
        "variation query 2",
        "variation query 3"
    ]
}}"""
    
    try:
        completion = await openai_client.chat.completions.create(
            model=QUERY_REWRITING_MODEL,
            messages=[
                {"role": "system", "content": expansion_prompt.format(query=query)}
            ],
            temperature=QUERY_REWRITING_TEMPERATURE,
            response_format={"type": "json_object"},
            max_tokens=500
        )
        result = json.loads(completion.choices[0].message.content)
        queries = result.get("queries", [query])
        
        # Ensure distinct queries and include original if not present (though prompt asks for variations)
        # It's often good to keep the original or a very close rewrite.
        # But here we trust the LLM to give 3 good ones.
        return queries[:3]
        
    except Exception as e:
        logger.exception("Error expanding query with domain knowledge:")
        return [query]  # Return original query as a single-item list if expansion fails

async def handle_clarification_response(original_query: str, clarification_response: str, 
                                     language: str = "English") -> dict:
    """
    Handle user's response to a clarification request and create an enhanced query.
    
    Args:
        original_query: The user's original question
        clarification_response: User's response to clarification
        language: Language of the queries
        
    Returns:
        Dictionary with enhanced query and action
    """
    try:
        # Combine the original query with clarification response
        combined_query = await combine_queries(original_query, clarification_response, language)
        
        # Expand the combined query with domain knowledge
        enhanced_queries = await expand_query_with_domain_knowledge(combined_query)
        
        return {
            "action": "rewrite",
            "rewritten_queries": enhanced_queries,
            "rewritten_query": enhanced_queries[0] if enhanced_queries else combined_query,
            "relevant_history_indices": [],
            "query_type": "Follow-up"
        }
        
    except Exception as e:
        logger.exception("Error handling clarification response:")
        # Fallback to simple combination
        combined_q = f"{original_query} {clarification_response}".strip()
        return {
            "action": "rewrite",
            "rewritten_queries": [combined_q],
            "rewritten_query": combined_q,
            "relevant_history_indices": [],
            "query_type": "Follow-up"
        }

async def combine_queries(original_query: str, clarification_response: str, 
                         language: str = "English") -> str:
    """
    Intelligently combine original query with clarification response.
    
    Args:
        original_query: User's original question
        clarification_response: User's response to clarification
        language: Language of the queries
        
    Returns:
        Enhanced combined query
    """
    try:
        combination_prompt = f"""You are an expert at combining user queries for better information retrieval.

**Original Query:** {original_query}
**Clarification Response:** {clarification_response}
**Language:** {language}

Create an enhanced, specific query that incorporates both the original question and the clarification details.
The combined query should:
- Include all relevant details from both queries
- Be specific and searchable
- Maintain the original intent
- Add necessary context for better retrieval

Enhanced query:"""

        completion = await openai_client.chat.completions.create(
            model=QUERY_REWRITING_MODEL,
            messages=[
                {"role": "system", "content": combination_prompt}
            ],
            temperature=QUERY_REWRITING_TEMPERATURE,
            max_tokens=300
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        logger.exception("Error combining queries:")
        # Simple fallback combination
        return f"{original_query} {clarification_response}".strip()