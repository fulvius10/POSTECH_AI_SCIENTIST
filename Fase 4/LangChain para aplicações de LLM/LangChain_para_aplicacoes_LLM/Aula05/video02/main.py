import operator
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

MODELO_OPENAI_PADRAO = os.environ.get("OPENAI_CHAT_MODEL")
if not MODELO_OPENAI_PADRAO:
    raise RuntimeError("Configure OPENAI_CHAT_MODEL no .env")

CAMINHO_BANCO = Path(__file__).parent / "memoria_grafo_aula5.db"

MAX_TENTATIVAS = 3

def criar_llm():
    return ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", MODELO_OPENAI_PADRAO))

class EstadoRedacao(TypedDict):
    tema: str
    rascunho: str
    critica: str
    aprovado: bool
    tentativas: int
    trilha: Annotated[list[str], operator.add]

def criar_app(llm, checkpointer):
    def gerar(estado: EstadoRedacao) -> dict:
        """Escreve (ou reescreve, se já houve crítica)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Escreva UMA frase de resumo executivo sobre o tema. "
             "Se houver crítica anterior, corrija exatamente o que ela apontou."),
            ("human", "Tema: {tema}\nCrítica anterior: {critica}"),
        ])
        texto = (prompt | llm | StrOutputParser()).invoke({
            "tema": estado["tema"],
            "critica": estado["critica"] or "(nenhuma, é a primeira versão)",
        })
        tentativa = estado["tentativas"] + 1
        return {
            "rascunho": texto,
            "tentativas": tentativa,
            "trilha": [f"gerar (tentativa {tentativa})"],
        }

    def revisar(estado: EstadoRedacao) -> dict:
        """O agente critica o próprio trabalho. É a autocrítica do loop."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Você revisa resumos executivos. Se o texto tiver UMA frase e citar "
             "um número, responda exatamente APROVADO. Caso contrário, responda "
             "com uma frase dizendo o que corrigir."),
            ("human", "{rascunho}"),
        ])
        parecer = (prompt | llm | StrOutputParser()).invoke(
            {"rascunho": estado["rascunho"]}
        ).strip()

        aprovado = parecer.upper().startswith("APROVADO")
        return {
            "critica": "" if aprovado else parecer,
            "aprovado": aprovado,
            "trilha": [f"revisar -> {'aprovado' if aprovado else 'reprovado'}"],
        }

    def publicar(estado: EstadoRedacao) -> dict:
        """Nó de alta criticidade: é aqui que o grafo vai pausar."""
        return {"trilha": [f"publicar -> {estado['rascunho'][:40]}..."]}

    def decidir(estado: EstadoRedacao) -> str:
        """ARESTA CONDICIONAL: calcula o próximo nó lendo o estado.

        Note o teto de tentativas. Sem ele, um revisor exigente demais deixa
        o grafo girando para sempre — e queimando tokens.
        """
        if estado["aprovado"]:
            return "publicar"
        if estado["tentativas"] >= MAX_TENTATIVAS:
            return "publicar"   # desiste de refinar e segue com o que tem
        return "gerar"          # <- a aresta que volta para trás: o ciclo

    grafo = StateGraph(EstadoRedacao)
    grafo.add_node("gerar", gerar)
    grafo.add_node("revisar", revisar)
    grafo.add_node("publicar", publicar)

    grafo.add_edge(START, "gerar")
    grafo.add_edge("gerar", "revisar")
    grafo.add_conditional_edges("revisar", decidir, ["gerar", "publicar"])
    grafo.add_edge("publicar", END)

    # interrupt_before: o grafo para ANTES de executar 'publicar' e devolve o
    # controle. Nada é publicado sem alguém olhar.
    return grafo.compile(checkpointer=checkpointer, interrupt_before=["publicar"])


def run_aula_5_video_2():
    print("--- AULA 5 / VÍDEO 02: CICLO, PERSISTÊNCIA E HUMAN-IN-THE-LOOP ---")

    llm = criar_llm()
    configuracao = {"configurable": {"thread_id": "redacao_q1"}}
    entrada = {
        "tema": "queda de 8% no faturamento do trimestre",
        "rascunho": "", "critica": "", "aprovado": False,
        "tentativas": 0, "trilha": [],
    }

    with SqliteSaver.from_conn_string(str(CAMINHO_BANCO)) as checkpointer:
        app = criar_app(llm, checkpointer)

        print("\n=== 1. O desenho do grafo (repare na aresta que volta) ===")
        print(app.get_graph().draw_mermaid())

        print("\n=== 2. Rodando até a pausa ===")
        for parcial in app.stream(entrada, config=configuracao, stream_mode="updates"):
            for no, mudanca in parcial.items():
                if no == "__interrupt__":       # o grafo bateu no interrupt_before
                    print("  [PAUSA] interrompido antes de 'publicar'")
                    continue
                print(f"  [{no}] {mudanca['trilha'][0]}")

        print("\n=== 3. O grafo PAUSOU. Estado na mão do humano ===")
        estado = app.get_state(configuracao)
        print(f"  Próximo nó a executar: {estado.next}")
        print(f"  Tentativas até aqui:   {estado.values['tentativas']}")
        print(f"  Rascunho proposto:     {estado.values['rascunho']}")

        print("\n=== 4. OVERRIDE: o humano edita o estado antes de liberar ===")
        app.update_state(
            configuracao,
            {"rascunho": estado.values["rascunho"] + " (revisado por Adriano)"},
        )
        print(f"  Rascunho após edição:  {app.get_state(configuracao).values['rascunho']}")

        print("\n=== 5. APPROVAL: invoke(None) retoma de onde parou ===")
        final = app.invoke(None, config=configuracao)
        print("  Trilha completa:")
        for passo in final["trilha"]:
            print(f"    - {passo}")

        print("\n=== 6. TIME TRAVEL: os snapshots salvos pelo checkpointer ===")
        historico = list(app.get_state_history(configuracao))
        print(f"  {len(historico)} snapshots gravados (do mais recente ao início):")
        for snap in historico[:6]:
            proximo = snap.next or ("FIM",)
            print(f"    próximo={proximo[0]:<12} tentativas={snap.values.get('tentativas')}")
        print("\n  Cada um desses tem um checkpoint_id: dá para reexecutar a partir")
        print("  de qualquer ponto passando-o no config.")

if __name__ == "__main__":

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERRO: configure OPENAI_API_KEY no .env antes de rodar o script.")
    else:
        run_aula_5_video_2()