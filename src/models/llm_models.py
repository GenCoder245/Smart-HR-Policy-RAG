from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import Settings

primary_llm: ChatOpenAI
fallback_llm: ChatGoogleGenerativeAI

def get_language_model(llm_settings: Settings):
    primary_llm = ChatOpenAI(model = llm_settings.openai_chat_model,
                            temperature = llm_settings.llm_temperature ,
                            api_key = llm_settings.openai_api_key,
                            max_retries = llm_settings.max_llm_retries,
                            )

    fallback_llm = ChatGoogleGenerativeAI(model = llm_settings.gemini_chat_model,
                                        temperature = llm_settings.llm_temperature ,
                                        api_key = llm_settings.google_api_key,
                                        max_retries = llm_settings.max_llm_retries,
                                        )

    llm_with_fallback = primary_llm.with_fallbacks(fallbacks= [fallback_llm])

    return llm_with_fallback


