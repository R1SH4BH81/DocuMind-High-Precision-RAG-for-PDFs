from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MODEL_NAME = "gemini-2.5-flash"
LOCAL_MODEL_PATH = ""


def get_cloud_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def get_local_llm():
    llm = LlamaCpp(
        model_path=LOCAL_MODEL_PATH,
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=32,
        temperature=0.7
    )
    return llm


def get_llm(use_cloud: bool):
    if use_cloud:
        return get_cloud_llm()
    else:
        return get_local_llm()
