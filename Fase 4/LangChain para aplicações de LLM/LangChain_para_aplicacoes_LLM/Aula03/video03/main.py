import os
from pathlib import Path
import re
import sqlite3
from contextlib import closing

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

# Fallback opcional: só é usado se o pacote estiver instalado.
try:
    from langchain_openai import ChatOpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

MODELO_CHAT_PADRAO = os.environ.get("GEMINI_CHAT_MODEL", "")
MODELO_OPENAI_PADRAO = os.environ.get("OPENAI_CHAT_MODEL", "")


# Arquivo único no diretório do projeto. Dá para abrir no VS Code e ver as linhas.
DATABASE = os.environ.get("SQLITE_PATH", "memoria_chat.db")
CAMINHO_BANCO = Path(Path(__file__).parent, DATABASE)
CAMINHO_BANCO.touch(exist_ok=True)

TABELA_HISTORICO = "historico_conversas"

TIPOS_PERSISTIDOS = {"human": HumanMessage, "ai": AIMessage}


def _mascarar_segredos(texto: str) -> str:
    """Remove credenciais que possam aparecer em mensagens de erro antes de imprimir."""
    return re.sub(r"(key=|AIza|sk-)[A-Za-z0-9_\-]{6,}", r"\1<REDACTED>", str(texto))


def _validar(llm, rotulo: str):
    """Faz uma chamada mínima para confirmar que o provedor responde.

    É aqui que a cota estourada aparece (429), em vez de estourar no meio da
    conversa depois de já ter gravado o turno do usuário no banco.
    """
    try:
        llm.invoke("ping")
        print(f" LLM OK: {rotulo}")
        return llm
    except Exception as e:
        print(f" {rotulo} indisponível -> {type(e).__name__}: {_mascarar_segredos(e)}")
        return None

def _criar_llm_google():
    if not os.environ.get("GOOGLE_API_KEY"):
        print(" GOOGLE_API_KEY não encontrada no .env.")
        return None
    modelo = os.environ.get("GOOGLE_CHAT_MODEL", MODELO_CHAT_PADRAO)
    return _validar(ChatGoogleGenerativeAI(model=modelo), f"Google / {modelo}")


def _criar_llm_openai():
    if not _HAS_OPENAI:
        print(" Pacote langchain-openai não instalado.")
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        print(" OPENAI_API_KEY não encontrada no .env.")
        return None
    modelo = os.environ.get("OPENAI_CHAT_MODEL", MODELO_OPENAI_PADRAO)
    return _validar(ChatOpenAI(model=modelo), f"OpenAI / {modelo}")


def criar_llm():
    """Escolhe o provedor do chat.

    LLM_PROVIDER aceita 'google', 'openai' ou 'auto' (padrão): tenta o Google e
    cai para a OpenAI se ele não responder — cota estourada, chave ausente ou
    modelo indisponível.
    """
    provedor = os.environ.get("LLM_PROVIDER", "auto").lower()

    if provedor == "google":
        return _criar_llm_google()
    if provedor == "openai":
        return _criar_llm_openai()

    return _criar_llm_google() or _criar_llm_openai()

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """Persistência de mensagens em SQLite.

    Abre uma conexão por operação e fecha em seguida. Uma conexão sqlite3 não
    pode ser usada em uma thread diferente da que a criou, e o LangChain executa
    a chain em uma thread pool — por isso nada de conexão guardada em atributo.
    Abrir conexão no SQLite é só abrir um arquivo local, então o custo é
    irrelevante (diferente do MySQL, onde a conexão é um socket de rede).
    """

    def __init__(self, session_id: str, caminho: str = None):
        self.session_id = session_id
        self.caminho = caminho or CAMINHO_BANCO
        self._garantir_tabela()

    def _conectar(self) -> sqlite3.Connection:
        conexao = sqlite3.connect(self.caminho, timeout=5.0)
        conexao.execute("PRAGMA journal_mode=WAL")  # evita 'database is locked'
        return conexao

    def _garantir_tabela(self) -> None:
        with closing(self._conectar()) as conexao, conexao:
            conexao.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABELA_HISTORICO} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('human', 'ai')),
                    content TEXT NOT NULL,
                    criado_em TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conexao.execute(
                f"CREATE INDEX IF NOT EXISTS idx_sessao ON {TABELA_HISTORICO} (session_id, id)"
            )

    @property
    def messages(self) -> list[BaseMessage]:
        with closing(self._conectar()) as conexao:
            linhas = conexao.execute(
                f"SELECT role, content FROM {TABELA_HISTORICO} "
                "WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            ).fetchall()

        return [
            TIPOS_PERSISTIDOS[role](content=content)
            for role, content in linhas
            if role in TIPOS_PERSISTIDOS
        ] 

    
    def add_messages(self, messages: list[BaseMessage]) -> None:
        linhas = [
            (self.session_id, m.type, m.content)
            for m in messages
            if m.type in TIPOS_PERSISTIDOS
        ]
        if not linhas:
            return

        # 'with conexao' abre transação: commit no sucesso, rollback na exceção.
        with closing(self._conectar()) as conexao, conexao:
            conexao.executemany(
                f"INSERT INTO {TABELA_HISTORICO} (session_id, role, content) VALUES (?, ?, ?)",
                linhas,
            )

    
    def clear(self) -> None:
        with closing(self._conectar()) as conexao, conexao:
            conexao.execute(
                f"DELETE FROM {TABELA_HISTORICO} WHERE session_id = ?",
                (self.session_id,),
            )

def run_aula_3_sqlite():
    print("--- INICIANDO AULA 3: MEMÓRIA PERSISTENTE COM SQLITE ---")
    print(f"Banco: {CAMINHO_BANCO}")

    llm = criar_llm()
    if llm is None:
        print("ERRO: nenhum provedor de LLM disponível. Verifique as chaves no .env.")
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente técnico de banco de dados. Seja breve e amigável."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    chain_com_memoria = RunnableWithMessageHistory(
        chain,
        get_session_history=SQLiteChatMessageHistory,
        input_messages_key="input",
        history_messages_key="history"
    )

    configuracao_usuario = {"configurable": {"session_id": "sessao_aluno_pos_tech"}}

    # ---- TURNO 1 ----
    print("\n[Turno 1] Enviando informações iniciais...")
    input_1 = "Olá professor! Meu nome é Lucas e estou desenvolvendo o projeto final em Python."
    print(f"Aluno: {input_1}")
    resposta_1 = chain_com_memoria.invoke({"input": input_1}, config=configuracao_usuario)
    print(f"Assistente: {resposta_1.content}")

    # ---- TURNO 2 (teste de retenção via banco) ----
    print("\n[Turno 2] Perguntando com base no histórico gravado no SQLite...")
    input_2 = "Com base no que eu te disse, qual é o meu nome e qual linguagem estou usando?"
    print(f"Aluno: {input_2}")
    resposta_2 = chain_com_memoria.invoke({"input": input_2}, config=configuracao_usuario)
    print(f"Assistente: {resposta_2.content}")

    print(f"\n--- Concluído. Abra o arquivo {CAMINHO_BANCO} no VS Code para ver as linhas gravadas. ---")

if __name__ == "__main__":
    load_dotenv()

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("ERRO: configure GOOGLE_API_KEY ou OPENAI_API_KEY no .env antes de rodar o script.")
    else:
        try:
            run_aula_3_sqlite()
        except Exception as e:
            print(f"\nERRO na execução: {type(e).__name__}: {_mascarar_segredos(e)}")
            