from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MODEL_NAME = "gemini-2.5-flash"
LOCAL_MODEL_NAME = "llama3.2:3b"


def get_cloud_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def get_local_llm():
    return ChatOllama(
        model=LOCAL_MODEL_NAME,
        temperature=0.0,
    )


def get_llm(use_cloud: bool):
    if use_cloud:
        return get_cloud_llm()
    else:
        return get_local_llm()
