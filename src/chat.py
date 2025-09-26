import os
import argparse
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from search import search_prompt

load_dotenv()
for k in ("OPENAI_API_KEY", "PGVECTOR_URL","PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

# Configurar embeddings, store e LLM uma vez
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

# Configurar LLM e prompt
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
prompt_template = search_prompt()
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def search_and_answer(query):
    """Executa busca e gera resposta usando IA"""
    # Buscar documentos relevantes
    results = store.similarity_search(query, k=3)
    
    if not results:
        return "Não tenho informações necessárias para responder sua pergunta."
    
    # Combinar contexto dos documentos encontrados
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Gerar resposta usando o prompt template
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    
    return response.content

def interactive_mode():
    """Modo interativo do CLI"""
    print("=== PDF Search CLI ===")
    print("Digite suas perguntas ou 'sair' para encerrar.")
    print("-" * 50)

    while True:
        try:
            # Input do usuário
            query = input("\nFaça sua pergunta: ").strip()
            
            # Comandos de saída
            if query.lower() in ['sair', 'exit', 'quit', '']:
                print("Encerrando. Até logo!")
                break
            
            # Resposta da IA
            print(f"\nPERGUNTA: {query}")
            response = search_and_answer(query)
            print(f"RESPOSTA: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nEncerrando...")
            break
        except Exception as e:
            print(f"Erro: {str(e)}")

def single_question_mode(question):
    """Modo de pergunta única"""
    try:
        response = search_and_answer(question)
        print(f"PERGUNTA: {question}")
        print(f"RESPOSTA: {response}")
    except Exception as e:
        print(f"Erro: {e}")

def main():
    parser = argparse.ArgumentParser(description="PDF Search CLI")
    parser.add_argument("-q", "--question", help="Fazer uma pergunta específica")
    parser.add_argument("-i", "--interactive", action="store_true", help="Modo interativo")
    
    args = parser.parse_args()
    
    if args.question:
        single_question_mode(args.question)
    elif args.interactive:
        interactive_mode()
    else:
        # Modo padrão interativo
        interactive_mode()

if __name__ == "__main__":
    main()