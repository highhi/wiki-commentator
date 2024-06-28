import os
import sys
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.chat_models import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def read_html_file(file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_name}' が見つかりません。")
        return ""
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に問題が発生しました: {e}")
        return ""


def process_wikipedia_text(file_name: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    raw_text = read_html_file(file_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


def initialize_bedrock_models():
    chat_model = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={
            "temperature": 0.01,
            "max_tokens": 500
        },
        region_name="us-east-1"
    )
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )
    return chat_model, embeddings


def create_or_load_vector_store(chunks: List[str], embeddings, file_path: str = "./vectorstore") -> FAISS:
    if os.path.exists(file_path):
        """ベクトルストアがあればロードする"""
        vector_store = FAISS.load_local(
            file_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(file_path)
    return vector_store


def perform_similarity_search(vector_store: FAISS, query: str, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return results


def setup_qa_chain(chat_model: ChatBedrock, vector_store: FAISS) -> Runnable:
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template("""Human: 与えられた質問に答えるAIです。

    {context}

    <text>{input}</text>
    
    Assistant:""")

    document_chain = create_stuff_documents_chain(chat_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def main():
    file_name = "wikipedia.html"
    processed_chunks = process_wikipedia_text(file_name)

    print(f"処理されたチャンクの数: {len(processed_chunks)}")

    chat_model, embeddings = initialize_bedrock_models()
    vector_store = create_or_load_vector_store(processed_chunks, embeddings)
    qa_chain = setup_qa_chain(chat_model, vector_store)

    try:
        while True:
            question = input("質問を入力してください（終了するには 'q' を入力）")

            if question.lower() == "q":
                print("終了します。")
                break

            if question.strip() == "":
                print("質問が入力されていません。もう一度入力してください。")
                continue

            print("回答を生成中...", end='', flush=True)

            response = qa_chain.invoke({"input": question})

            print("\r" + " "*20 + "\r", end='')

            print("\n回答:")
            print(response["answer"])

            print("\n")
    except KeyboardInterrupt:
        print("\nプログラムが中断されました。終了します。")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
