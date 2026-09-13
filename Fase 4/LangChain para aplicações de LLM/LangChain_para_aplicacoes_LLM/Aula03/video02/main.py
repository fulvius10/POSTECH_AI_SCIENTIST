import os
import re

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

MODELO_EMBEDDING_PADRAO = "gemini-embedding-001"
MODELO_CHAT_PADRAO = "gemini-3.5-flash"
MODELO_OPENAI_PADRAO = os.environ.get("OPENAI_CHAT_MODEL", "")

QTD_VARIACOES = 2
DOCS_POR_CONSULTA = 2

def _mascarar_segredos(texto: str) -> str:
    """Remove chaves de API que possam aparecer em mensagens de erro antes de imprimir."""
    return re.sub(r"(key=|AIza)[A-Za-z0-9_\-]{6,}", r"\1<REDACTED>", str(texto))


def _nomes_candidatos(modelo: str):
    """Nome puro e com prefixo 'models/'.

    langchain-google-genai >= 3.x normaliza sozinho; a 2.x manda a string crua
    para a API, que exige o prefixo e devolve 'unexpected model name format'.
    """
    if modelo.startswith("models/"):
        return [modelo, modelo.removeprefix("models/")]
    return [modelo, f"models/{modelo}"]

def criar_embeddings():
    """Instancia e valida os embeddings do Gemini antes de indexar qualquer coisa."""
    modelo = os.environ.get("GOOGLE_EMBEDDING_MODEL", MODELO_EMBEDDING_PADRAO)

    for nome in _nomes_candidatos(modelo):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=nome)
            embeddings.embed_query("ping")  # valida credencial + nome do modelo
            print(f" Embeddings OK (modelo: {nome})")
            return embeddings
        except Exception as e:
            print(f" Modelo '{nome}' indisponível -> {type(e).__name__}: {_mascarar_segredos(e)}")

    return None

def _extrair_variacoes(texto: str, limite: int):
    """Transforma a resposta da LLM em uma lista de perguntas, sem numeração nem linhas vazias."""
    variacoes = []
    for linha in texto.splitlines():
        limpa = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", linha).strip()
        if limpa:
            variacoes.append(limpa)
    return variacoes[:limite]

def buscar_multi_query(vector_store, consultas, k: int):
    """Busca cada consulta e devolve a união dos documentos, sem repetidos.

    O dict preserva a ordem de inserção, então os documentos da pergunta
    original aparecem primeiro no contexto.
    """
    unicos = {}
    for consulta in consultas:
        for doc in vector_store.similarity_search(consulta, k=k):
            unicos.setdefault(doc.page_content, doc)
    return list(unicos.values())

def run_aula_2():
    print("--- INICIANDO AULA 2: ADVANCED RAG (MULTI-QUERY) ---")

    embeddings = criar_embeddings()
    if embeddings is None:
        print("ERRO: não foi possível inicializar os embeddings. Verifique GOOGLE_API_KEY no .env.")
        return

    llm = ChatGoogleGenerativeAI(model=os.environ.get("GOOGLE_CHAT_MODEL", MODELO_CHAT_PADRAO))

    # 1. Base de conhecimento local pré-indexada
    textos = [
        "O faturamento consolidado da ACME em Maio de 2026 foi de R$ 12.5 milhões.",
        "A divisão de Microcontroladores teve expansão de 15% na operação global.",
        "O Diretor Executivo da ACME em 2026 é o Engenheiro Carlos Eduardo."
    ]

    try:
        vector_store = FAISS.from_texts(textos, embeddings)
    except Exception as e:
        print("ERRO ao indexar os vetores:", _mascarar_segredos(e))
        return

    # 2. Query Transformation: a LLM gera variações para aumentar o Recall
    pergunta_usuario = "Quem lidera a empresa ACME e como foi o lucro?"
    print(f"\n[Pergunta Original]: {pergunta_usuario}")

    prompt_multi_query = PromptTemplate.from_template(
        "Você é um assistente técnico de IA. Gere {qtd} variações alternativas da seguinte pergunta "
        "para ajudar na busca em um banco de dados vetorial.\n"
        "Responda APENAS com as perguntas, uma por linha, sem numeração e sem comentários.\n"
        "Pergunta original: {pergunta}"
    )

    chain_variacoes = prompt_multi_query | llm | StrOutputParser()

    try:
        resposta_variacoes = chain_variacoes.invoke(
            {"pergunta": pergunta_usuario, "qtd": QTD_VARIACOES}
        )
    except Exception as e:
        print("ERRO ao gerar variações:", _mascarar_segredos(e))
        return

    variacoes = _extrair_variacoes(resposta_variacoes, QTD_VARIACOES)

    print("\n[Query Transformation - Variações Geradas pela LLM]:")
    for v in variacoes:
        print(f" - {v}")

    # 3. Retrieval: busca a pergunta original E cada variação, unindo os resultados
    print("\n[Retrieval] Buscando informações relevantes...")
    consultas = [pergunta_usuario] + variacoes
    documentos_recuperados = buscar_multi_query(vector_store, consultas, k=DOCS_POR_CONSULTA)

    contexto = "\n".join(doc.page_content for doc in documentos_recuperados)
    print(f"{len(documentos_recuperados)} documento(s) único(s) recuperado(s):\n{contexto}")

    # 4. Generation: resposta fundamentada apenas no contexto recuperado
    prompt_rag = PromptTemplate.from_template(
        "Você é um assistente corporativo. Responda à pergunta do usuário utilizando APENAS o contexto fornecido abaixo.\n"
        "Se não souber a resposta com base no contexto, diga honestamente que não encontrou a informação.\n\n"
        "Contexto:\n{contexto}\n\n"
        "Pergunta: {pergunta}\n\n"
        "Resposta Resumida e Direta:"
    )

    chain_rag = prompt_rag | llm | StrOutputParser()

    try:
        resposta_final = chain_rag.invoke({"contexto": contexto, "pergunta": pergunta_usuario})
    except Exception as e:
        print("ERRO ao gerar a resposta final:", _mascarar_segredos(e))
        return

    print("\n[Resposta Final Fundamentada da LLM]:")
    print(resposta_final)

if __name__ == "__main__":
    load_dotenv()
    run_aula_2()