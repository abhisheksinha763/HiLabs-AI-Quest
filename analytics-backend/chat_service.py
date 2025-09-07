import ollama
from typing import List, Dict, Generator
import logging

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, chat_model: str = "llama3.1"):
        self.chat_model = chat_model
        self.system_prompt = """You are a helpful assistant that must answer strictly using the provided CSV context about healthcare providers. 
If the answer is not in the context, say you don't know. Be concise and cite which chunk(s) you used like [C1], [C2].
Focus on provider information, licenses, locations, and compliance data."""
    
    def _make_messages(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Create messages for the chat model with context."""
        context_block_lines = []
        for c in retrieved_chunks:
            context_block_lines.append(f"[C{c['chunk_id']}]")
            context_block_lines.append(c["text"])
            context_block_lines.append("")
        
        context_block = "\n".join(context_block_lines).strip()
        user_prompt = f"""Answer the question using only this healthcare provider CSV context:

{context_block}

Question: {query}
"""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def chat_stream(self, query: str, retrieved_chunks: List[Dict]) -> Generator[str, None, None]:
        """Stream chat response using retrieved context."""
        messages = self._make_messages(query, retrieved_chunks)
        
        try:
            stream_resp = ollama.chat(model=self.chat_model, messages=messages, stream=True)
            for part in stream_resp:
                chunk = part.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Error in chat streaming: {e}")
            yield f"\n[Error] Failed to get response from chat model: {str(e)}"
    
    def chat_complete(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Get complete chat response (non-streaming)."""
        messages = self._make_messages(query, retrieved_chunks)
        
        try:
            response = ollama.chat(model=self.chat_model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return f"[Error] Failed to get response from chat model: {str(e)}"
