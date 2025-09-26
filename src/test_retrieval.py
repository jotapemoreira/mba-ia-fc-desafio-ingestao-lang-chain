import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from search import search_prompt

load_dotenv()

def test_retrieval():
    # Embeddings
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))
    
    # Vector store usando a coleção correta
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )
    
    # Testar busca genérica primeiro
    query = "copa do mundo"
    print(f"Testando busca por: '{query}'")
    
    # Busca por similaridade
    docs = vectorstore.similarity_search(query, k=3)
    print(f"Documentos encontrados: {len(docs)}")
    
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Relevante")
        print(f"Conteúdo: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    
    # Testar busca específica
    query2 = "1994"
    print(f"\n\nTestando busca por: '{query2}'")
    docs2 = vectorstore.similarity_search(query2, k=3)
    print(f"Documentos encontrados: {len(docs2)}")
    
    for i, doc in enumerate(docs2, 1):
        print(f"\n{i}. Relevante")
        print(f"Conteúdo: {doc.page_content[:200]}...")

if __name__ == "__main__":
    test_retrieval()