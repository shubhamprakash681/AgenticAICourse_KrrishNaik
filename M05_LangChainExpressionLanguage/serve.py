from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_key)

# Create Chat Template
chat_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "Translate the following into {language}:"},
    {"role": "user", "content": "{text}"}
])

# Create Output Parser
output_parser = StrOutputParser()

# Chain
chain = chat_prompt | model | output_parser

# Pydantic model for input
class TranslationRequest(BaseModel):
    language: str
    text: str

# App
app = FastAPI(
    title="Langchain Groq Translation API", 
    description="A simple API server using Langchain Runnable Interface to translate texts into desired language.",
    version="1.0"
)

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        result = chain.invoke({
            "language": request.language,
            "text": request.text
        })
        return {"translation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", log_level="info", port=3000)