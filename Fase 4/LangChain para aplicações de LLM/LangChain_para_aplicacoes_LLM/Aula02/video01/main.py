from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

# CONFIGURAÇÃO DE AMBIENTE
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# 1. Configuração das credenciais (Mantenha suas chaves seguras)
# Validação para garantir que a chave de API foi carregada
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("A variável de ambiente para a API_KEY não foi encontrada. Crie um arquivo .env e adicione-a.")

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
parser = StrOutputParser()

# 2. Criação dos Componentes da Chain
prompt_gerador = ChatPromptTemplate.from_template(
    "Gere uma breve descrição técnica sobre o conceito de: {topico}"
)

# 3. Função customizada usando RunnableLambda (Sanitização)
prompt_revisor = ChatPromptTemplate.from_template(
    "Melhore o seguinte texto técnico, corrigindo eventuais termos e tornando-o mais formal:\n\n{texto_gerado}"
)

def limpar_texto(texto: str) -> str:
    # Simula uma limpeza ou remoção de caracteres indesejados
    return texto.strip().replace("  ", " ")

runnable_limpeza = RunnableLambda(limpar_texto)

# 4. Construindo a Chain com o Operador Pipe (|)
chain = (
    prompt_gerador
    | model 
    | parser 
    | runnable_limpeza
    | {"texto_gerado": RunnablePassthrough()}  # Mantém o dado vivo no dicionário para o próximo prompt
    | prompt_revisor 
    | model 
    | parser
)

print("=== 1. DEMONSTRANDO .invoke() ===")
resultado_invoke = chain.invoke({"topico": "LangChain Expression Language (LCEL)"})
print(resultado_invoke)
print("\n" + "="*40 + "\n")

print("=== 2. DEMONSTRANDO .stream() ===")
# Excelente para mostrar o surgimento de tokens em tempo real na gravação
for chunk in chain.stream({"topico": "RunnableParallel"}):
    print(chunk, end="", flush=True)
print("\n\n" + "="*40 + "\n")

print("=== 3. DEMONSTRANDO .batch() ===")
# Execução paralela nativa de múltiplos inputs
inputs = [
    {"topico": "Large Language Models"},
    {"topico": "API Idempotency"}
]

resultados_batch = chain.batch(inputs)
for i, res in enumerate(resultados_batch):
    print(f"Resultado {i+1}:\n{res}\n")

    