import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Silencia avisos de schema não suportado para limpar o console
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

load_dotenv()

print("--- Configurando um Agente com Memória e Ferramentas Customizadas ---")

@tool
def calcular_custo_tokens(quantidade_tokens: int, modelo: str) -> str:
    """Calcula o custo estimado em dólares com base na quantidade de tokens e no modelo especificado."""
    modelo = modelo.lower()
    if "gpt-4o-mini" in modelo:
        custo = (quantidade_tokens / 1_000_000) * 0.15 # Exemplo de preço hipotético por milhão
        return f"O custo estimado para {quantidade_tokens} tokens no gpt-4o-mini é de ${custo:.6f} USD."
    else:
        return "Modelo não mapeado para cálculo de custo local."

# Lista de ferramentas disponíveis para o agente
tools = [calcular_custo_tokens]

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

# Criando o Prompt do Agente contendo um espaço reservado para o histórico (Memória)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um agente financeiro de IA. Use as ferramentas disponíveis para responder dúvidas sobre custos se necessário."),
    MessagesPlaceholder(variable_name="chat_history"), # Onde o histórico da conversa reside
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Onde o agente "pensa" e planeja as ações
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

historico_conversa = [
    HumanMessage(content="Olá, meu nome é Carlos e estou construindo um assistente."),
    AIMessage(content="Olá Carlos! Como posso te ajudar com os custos ou arquitetura do seu assistente hoje?")
]

pergunta_usuario = "Se meu sistema rodar 5.000.000 de tokens no gpt-4o-mini, quanto vou gastar?"

print(f"\n[Usuário]: {pergunta_usuario}")
resposta_final = agent_executor.invoke({
    "input": pergunta_usuario,
    "chat_history": historico_conversa
})

print("\n=== RESPOSTA FINAL DO AGENTE ===")
print(resposta_final["output"])
