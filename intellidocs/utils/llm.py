from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("LLM not found ")
else:
    genai.configure(api_key=GEMINI_API_KEY)

def call_gemini(query: str, chunks: list) -> Dict[str, Any]:
    """
    Call Gemini API with query and relevant document chunks
    
    Args:
        query: User's question
        chunks: List of relevant text chunks from documents
        
    Returns:
        Structured JSON response
    """
    if not GEMINI_API_KEY:
        return {
            "error": "LLM not configured",
            "decision": "error",
            "confidence": 0.0
        }

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        # Create context from chunks
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])

        # Enhanced insurance decision-making prompt
        prompt = f"""
You are an AI assistant for an insurance company. Your job is to determine if a claim should be **accepted** or **rejected** based on insurance policy documents.

DOCUMENT CONTEXT:
{context}

USER QUERY (claim scenario):
"{query}"

Please analyze the document context and return a structured response strictly in the following JSON format:

{{
  "decision": "<accepted/rejected/pending/unknown>",
  "justification": [
    {{
      "clause": "<specific clause or section>",
      "text": "<exact supporting text from the document>",
      "relevance": "<why this text supports the decision>"
    }}
  ],
  "confidence": <float between 0.0 and 1.0>,
  "summary": "<short human-readable summary of your reasoning>",
  "reasoning": "<step-by-step explanation of how the decision was made>"
}}

RULES:
- Only use the content from the DOCUMENT CONTEXT.
- Do NOT assume or hallucinate information.
- Be concise and structured.
- If the document doesn't support a clear decision, use "pending" or "unknown".

Return only valid JSON. Do not include any extra commentary.
"""

        logger.info("Sending request to LLM...")
        response = model.generate_content(prompt)

        if not response.text:
            return {
                "error": "Empty response from LLM",
                "decision": "error",
                "confidence": 0.0
            }

        # Parse JSON response
        try:
            text = response.text.strip()
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                return create_fallback_response(query, text)

            json_text = text[json_start:json_end]
            parsed_response = json.loads(json_text)

            # Validate required fields
            required_fields = ["decision", "confidence"]
            for field in required_fields:
                if field not in parsed_response:
                    parsed_response[field] = "unknown" if field == "decision" else 0.5

            # Ensure confidence is float
            try:
                confidence = float(parsed_response.get("confidence", 0.5))
                parsed_response["confidence"] = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                parsed_response["confidence"] = 0.5

            logger.info(f"Successfully parsed LLM response with confidence: {parsed_response['confidence']}")
            return parsed_response

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return create_fallback_response(query, response.text)

    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return {
            "error": f"Failed to process query: {str(e)}",
            "decision": "error",
            "confidence": 0.0,
            "raw_response": getattr(response, 'text', '') if 'response' in locals() else ''
        }

def create_fallback_response(query: str, raw_text: str) -> Dict[str, Any]:
    """Create a structured response when JSON parsing fails"""
    return {
        "decision": "processed",
        "confidence": 0.5,
        "justification": [{
            "clause": "General Analysis",
            "text": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
            "relevance": "AI-generated fallback response to the query"
        }],
        "summary": f"Processed query: {query}",
        "reasoning": "Fallback used due to parsing failure",
        "raw_response": raw_text,
        "note": "This is a fallback response due to invalid or unstructured LLM output"
    }
