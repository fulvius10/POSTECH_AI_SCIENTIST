import os
import re

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Tenta usar embeddings do Google (Gemini); usa OpenAI como fallback se não disponível
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    _HAS_GOOGLE_EMB = True
except Exception:
    _HAS_GOOGLE_EMB = False

try:
    from langchain_openai import OpenAIEmbeddings
    _HAS_OPENAI_EMB = True
except Exception:
    _HAS_OPENAI_EMB = False

MODELO_GEMINI_PADRAO = "gemini-embedding-001"

def _mascarar_segredos(texto: str) -> str:
    """Remove chaves de API que possam aparecer em mensagens de erro antes de imprimir."""
    return re.sub(r"(key=|AIza)[A-Za-z0-9_\-]{6,}", r"\1<REDACTED>", str(texto))

def _nomes_candidatos(modelo: str):
    """Gera as variações aceitas do nome do modelo.
 
    langchain-google-genai >= 3.x aceita o nome puro ('gemini-embedding-001');
    versões 2.x mandam a string crua para a API, que exige o prefixo 'models/'
    e devolve 'unexpected model name format' sem ele. Tentar as duas cobre
    ambos os casos sem depender da versão instalada.
    """
    if modelo.startswith("models/"):
        return [modelo, modelo.removeprefix("models/")]
    return [modelo, f"models/{modelo}"]

def _criar_embeddings_google():
    """Instancia e valida os embeddings do Gemini. Retorna None se não der para usar."""
    if not _HAS_GOOGLE_EMB:
        print(" Pacote langchain-google-genai não instalado.")
        return None
    if not os.environ.get("GOOGLE_API_KEY"):
        print(" GOOGLE_API_KEY não encontrada no ambiente (.env).")
        return None
 
    modelo = os.environ.get("GOOGLE_EMBEDDING_MODEL", MODELO_GEMINI_PADRAO)
 
    for nome in _nomes_candidatos(modelo):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=nome)
            # Chamada mínima só para validar credencial + nome do modelo antes de indexar.
            embeddings.embed_query("ping")
            print(f" Usando GoogleGenerativeAIEmbeddings (modelo: {nome})")
            return embeddings
        except Exception as e:
            print(f" Modelo '{nome}' indisponível -> {type(e).__name__}: {_mascarar_segredos(e)}")
 
    return None

def _criar_embeddings_openai():
    """Fallback: embeddings da OpenAI. Retorna None se não der para usar."""
    if not _HAS_OPENAI_EMB or not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print(" Usando OpenAIEmbeddings como fallback")
        return embeddings
    except Exception as e:
        print(" Falha ao inicializar OpenAIEmbeddings:", _mascarar_segredos(e))
        return None

def criar_embeddings():
    """Escolhe o provedor de embeddings: Gemini primeiro, OpenAI como reserva."""
    return _criar_embeddings_google() or _criar_embeddings_openai()

def run_aula_1():
    print("--- INICIANDO AULA 3: PIPELINE DE INGESTÃO LOCAL ---")

    # 1. Dados corporativos simulados (Relatório Financeiro)
    documento_corporativo = """
    Relatório de Desempenho Financeiro - Indústrias ACME (Maio 2026).
    O faturamento consolidado no último mês atingiu R$ 12.5 milhões, representando uma alta de 8% em relação a abril.
    O principal motor de crescimento foi a divisão de Microcontroladores, que expandiu sua operação global em 15%.
    As despesas operacionais (OPEX) totalizaram R$ 4.2 milhões, mantendo-se dentro do orçamento previsto.
    A projeção para o próximo trimestre é manter a taxa de crescimento acima de dois dígitos.
    """

    # 2. Chunking (Fragmentação Inteligente) com Overlap para não perder contexto
    print("\n[Passo 1] Fragmentando o texto...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )
    chunks = text_splitter.split_text(documento_corporativo)
    for i, chunk in enumerate(chunks):
        print(f" Chunk {i+1}: {chunk.strip()}")

    # 3. Inicialização do modelo de Embedding
    print("\n[Passo 2] Gerando vetores (Gemini/Google ou OpenAI, conforme disponível)...")
    embeddings = criar_embeddings()
    if embeddings is None:
        print("ERRO: Nenhum provedor de embeddings disponível. "
              "Configure GOOGLE_API_KEY (ou OPENAI_API_KEY) e as bibliotecas necessárias.")
        return

    # 4. Criação do Banco Vetorial local com FAISS
    print("\n[Passo 3] Indexando vetores no FAISS...")
    try:
        vector_store = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        print("ERRO ao indexar os vetores:", _mascarar_segredos(e))
        return

    # 5. Teste de Busca Semântica (Simulando a pergunta do usuário)
    query = "Quanto a empresa faturou no mês passado?"
    print(f"\n[Passo 4] Executando busca semântica para: '{query}'")

    resultados = vector_store.similarity_search(query, k=1)

    print("\n[Resultado da Busca] Documento mais relevante encontrado:")
    print(f" -> {resultados[0].page_content.strip()}")

if __name__ == "__main__":
    load_dotenv()
    run_aula_1()