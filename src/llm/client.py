from typing import Any
from openai import OpenAI
from config.settings import settings
import structlog

logger = structlog.get_logger()


class LLMClient:
    """
    Reasoning-enabled OpenAI client wrapper.
    Modelin düşünme sürecini (reasoning_details) takip eder ve korur.
    """
    
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self.model = settings.llm_model
        self.conversation_history: list[dict[str, Any]] = []
        
    def chat(
        self,
        user_message: str,
        preserve_reasoning: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Reasoning-enabled chat completion.
        
        Args:
            user_message: Kullanıcı mesajı
            preserve_reasoning: Modelin düşünme sürecini koru (önerilen: True)
            extra_body: Ekstra body parametreleri
            
        Returns:
            {
                "content": str,  # Model cevabı
                "reasoning_details": Any,  # Düşünme süreci detayları
                "full_response": Any  # Tam OpenAI response objesi
            }
        """
        # Kullanıcı mesajını history'ye ekle
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # API çağrısı için body hazırla
        body = {"reasoning": {"enabled": True}}
        if extra_body:
            body.update(extra_body)
        
        try:
            # API çağrısı
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                extra_body=body,
            )
            
            # Assistant mesajını çıkar
            assistant_message = response.choices[0].message
            
            # Assistant mesajını history'ye ekle (reasoning_details ile birlikte)
            if preserve_reasoning and hasattr(assistant_message, 'reasoning_details'):
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "reasoning_details": assistant_message.reasoning_details
                })
            else:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
            
            logger.info(
                "LLM chat completed",
                content_length=len(assistant_message.content),
                has_reasoning=hasattr(assistant_message, 'reasoning_details')
            )
            
            return {
                "content": assistant_message.content,
                "reasoning_details": getattr(assistant_message, 'reasoning_details', None),
                "full_response": response
            }
            
        except Exception as e:
            logger.error("LLM chat failed", error=str(e), exc_info=True)
            raise
    
    def reset_conversation(self):
        """Conversation history'yi sıfırla"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Mevcut conversation history'yi döndür"""
        return self.conversation_history.copy()


# Singleton instance
llm_client = LLMClient()
