# Script_Video_03.py
import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

load_dotenv()

# Validação para garantir que a chave de API foi carregada
if not (os.getenv("GOOGLE_API_KEY") and os.getenv("TAVILY_API_KEY")):
    raise ValueError("A variável de ambiente GOOGLE_API_KEY e TAVILY_API_KEY não foram encontradas.\
                      Crie um arquivo .env e adicione-a.")

print("--- Iniciando o Projeto Prático: Answer Engine ---")

# 1. Instanciando uma ferramenta de busca em tempo real na internet
busca_internet = TavilyAnswer()
ferramentas_projeto = [busca_internet]

# 2. Inicializando a LLM que atuará como controladora
llm_controladora = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0
)

# 3. Definindo o prompt de comportamento do Answer Engine
prompt_engine = ChatPromptTemplate.from_messages([
    ("system", (
        "Você é o 'Answer Engine', um motor de respostas avançado integrado à internet. "
        "Sempre que o usuário perguntar sobre fatos recentes, ações de mercado, notícias ou cotações, "
        "use a ferramenta de busca para trazer dados verídicos e cite as descobertas de forma resumida."
    )),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Montando o agente focado em Tool Binding (vincular ferramentas ao modelo)
engine_agent = create_tool_calling_agent(llm_controladora, ferramentas_projeto, prompt_engine)

# 5. Criando o executor do Answer Engine
answer_engine_executor = AgentExecutor(
    agent=engine_agent, 
    tools=ferramentas_projeto, 
    verbose=True, # Exibe no terminal os passos de "Thought -> Action -> Observation"
    handle_parsing_errors=True # Garante resiliência caso o formato de saída quebre
)

# 6. Execução do teste prático (Uma pergunta factual recente que a LLM não saberia sem a ferramenta)
pergunta_atual = "Quem ganhou o último campeonato de Fórmula 1 e qual foi a pontuação final?"

print(f"\n[Consultando Answer Engine]: {pergunta_atual}")
resultado_engine = answer_engine_executor.invoke({"input": pergunta_atual})

print("\n=== RESPOSTA ENRICHED (COM BUSCA EM TEMPO REAL) ===")
print(resultado_engine["output"])
