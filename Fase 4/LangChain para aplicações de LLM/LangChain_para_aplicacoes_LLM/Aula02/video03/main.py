import os
import logging
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.globals import set_debug
from langchain_core.callbacks import BaseCallbackHandler

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
set_debug(False)

# Checagem básica de credenciais (não interrompe, apenas alerta)
if not os.getenv("OPENAI_API_KEY"):
    logging.warning("OPENAI_API_KEY não encontrada. Chamadas ao OpenAI vão falhar se não configurada.")
if not os.getenv("GOOGLE_API_KEY"):
    logging.warning("GOOGLE_API_KEY não encontrada. Chamadas ao Google Generative AI vão falhar se não configurada.")


# 1. Definição de uma Custom Tool com Tipagem e Docstring Ricas
@tool
def calcular_faturamento_anual(empresa: str, receita_mensal: float) -> str:
    """Calcula a projeção de faturamento anual de uma empresa baseada na receita mensal."""
    # A docstring funciona como a descrição lida pelo agente para decidir o uso da ferramenta
    faturamento_projetado = receita_mensal * 12
    return f"A projeção de faturamento anual para a empresa {empresa} é de R$ {faturamento_projetado:,.2f}"

class LogFallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        nome_modelo = serialized.get("id", ["LLM"])[-1]
        print(f"\n🟡 [TENTATIVA] Chamando modelo: {nome_modelo}")

    def on_llm_error(self, error, **kwargs):
        print(f"🔴 [FALHA] Modelo falhou com o erro: {error}")
        print("🔄 [FALLBACK] Redirecionando para o modelo secundário...")

    def on_llm_end(self, response, **kwargs):
        print("🟢 [SUCESSO] LLM respondeu com sucesso!")

tools = [calcular_faturamento_anual]

# 2. Prompt Padrão
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente financeiro corporativo de elite. Utilize as ferramentas fornecidas sempre que precisar de dados calculados."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. Modelos
llm_primario = ChatOpenAI(model="gpt-5.4-mini", temperature=0, timeout=0.001, max_retries=0)
llm_secundario = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. Criação dos Agentes Individuais com Tool Calling Universal
agent_primario = create_tool_calling_agent(llm_primario, tools, prompt)
agent_secundario = create_tool_calling_agent(llm_secundario, tools, prompt)

# 5. Aplicação do Fallback no Nível do Agente (Runnable)
agent_resiliente = agent_primario.with_fallbacks([agent_secundario])

# 6. Runtime Executor
agent_executor = AgentExecutor(
    agent=agent_resiliente,
    tools=tools,
    verbose=True
)

print("=== EXECUTANDO O AGENTE AUTÔNOMO ===")
consulta = "A empresa AlfaTech fatura 25000 reais por mês. Qual a projeção anual deles?"

try:
    resposta = agent_executor.invoke({"input": consulta}, config={"callbacks": [LogFallbackHandler()]})
    print("\n=== RESPOSTA FINAL DO AGENTE ===")
    # Normaliza formatos de saída comuns
    if isinstance(resposta, dict):
        out = resposta.get("output") or resposta.get("text") or resposta.get("result") or str(resposta)
    else:
        out = str(resposta)
    print(out)
except Exception as e:
    logging.exception("Falha na invocação do agente")
    print("Erro ao executar o agente:", e)
