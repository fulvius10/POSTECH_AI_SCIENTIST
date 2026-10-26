import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# CONFIGURAÇÃO DE AMBIENTE
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Validação para garantir que a chave de API foi carregada
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi encontrada. Crie um arquivo .env e adicione-a.")

print("--- Inicializando o Modelo e a Prompt Template ---")

# 2. Definindo o modelo de linguagem (LLM)
model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente técnico especialista em arquitetura de software de Inteligência Artificial. Seja direto e profissional."),
    ("user", "Explique brevemente o conceito de {topico} e por que ele é modular.")
])

output_parser = StrOutputParser()

chain = prompt_template | model | output_parser

topico_exemplo = "LangChain Expression Language (LCEL)"
print(f"\n[Executando Chain] Solicitando explicação sobre: {topico_exemplo}...")  

resposta = chain.invoke({"topico": topico_exemplo})

print("\n=== RESPOSTA DA CHAIN INTERNA ===")
print(resposta)