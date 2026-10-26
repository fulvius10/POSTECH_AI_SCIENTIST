import os
from pydantic import BaseModel, Field, field_validator
import unicodedata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

# Validação para garantir que a chave de API foi carregada
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("A variável de ambiente para a API_KEY não foi encontrada. Crie um arquivo .env e adicione-a.")

# 1. Configuração do Modelo de Classificação Inteligente (Pydantic)
class RoteadorSuporte(BaseModel):
    """Classifica a intenção do usuário para direcionar ao setor correto.

    Normaliza o valor de `setor` para lowercase antes da validação, assim
    aceitando respostas como "Financeiro" ou "financeiro" vindas do LLM.
    """
    setor: str = Field(
        description="O setor para onde a dúvida deve ser enviada baseado no contexto."
    )

    @field_validator("setor", mode="before")
    def _normalize_setor(cls, v):
        if isinstance(v, str):
            # Remove acentos e normalize para facilitar mapeamento
            s = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode("ascii")
            s = s.strip().lower()
            # Map common variations to canonical values
            if "finance" in s:
                return "financeiro"
            if "tecn" in s or "suporte" in s or "tecnico" in s:
                return "tecnico"
            return s
        return v

    @field_validator("setor")
    def _validate_setor(cls, v):
        if v not in ("financeiro", "tecnico"):
            raise ValueError("setor must be 'financeiro' or 'tecnico'")
        return v

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

llm_classificador = model.with_structured_output(RoteadorSuporte)

# 2. Definição das Sub-Chains Especializadas
prompt_financeiro = ChatPromptTemplate.from_template(
    "Você é um especialista financeiro. Responda a seguinte dúvida sobre faturamento de forma clara e cordial: {pergunta}"
)
prompt_tecnico = ChatPromptTemplate.from_template(
    "Você é um engenheiro de software sênior. Responda a seguinte dúvida de TI de forma técnica e objetiva: {pergunta}"
)

parser = StrOutputParser()

chain_financeira = prompt_financeiro | model | parser
chain_tecnica = prompt_tecnico | model | parser

# 3. Lógica do Roteador (Decisor Dinâmico)
def rotear_consulta(dados):
    intencao = dados["intencao"].setor
    pergunta_original = dados["pergunta"]
    
    print(f"\n[Roteador] -> Intenção detectada: {intencao.upper()}")
    
    if intencao == "financeiro":
        return chain_financeira.invoke({"pergunta": pergunta_original})
    elif intencao == "tecnico":
        return chain_tecnica.invoke({"pergunta": pergunta_original})

# 4. Orquestração da Chain de Roteamento Completa
# Usamos RunnableParallel de forma implícita passando o dicionário
chain_principal = {
    "intencao": ChatPromptTemplate.from_template("{pergunta}") | llm_classificador,
    "pergunta": RunnablePassthrough()
} | RunnableLambda(rotear_consulta)

# print("=== TESTE 1: INPUT FINANCEIRO ===")
# pergunta_1 = "Minha fatura veio duplicada este mês, como solicito o estorno?"
# resposta_1 = chain_principal.invoke({"pergunta": pergunta_1})
# print(f"Resposta Final:\n{resposta_1}")
# print("\n" + "="*40 + "\n")

print("=== TESTE 2: INPUT TÉCNICO ===")
pergunta_2 = "Estou recebendo erro 500 ao tentar conectar no banco PostgreSQL local."
resposta_2 = chain_principal.invoke({"pergunta": pergunta_2})
print(f"Resposta Final:\n{resposta_2}")