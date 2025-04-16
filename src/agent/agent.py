from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document, BaseRetriever
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
import traceback
import sys
import io
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pydantic import Field, BaseModel

# Set UTF-8 encoding for stdout and stdin
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Initialize LangSmith client
langsmith_client = Client()

# Initialize callback manager with LangSmith tracer
callback_manager = CallbackManager([LangChainTracer()])

# Initialize reranking model
reranker_model = None
reranker_tokenizer = None


def initialize_reranker():
    """Initialize the NVIDIA reranking model"""
    global reranker_model, reranker_tokenizer
    model_name = "nvidia/nemo-reranker-base"  # or nvidia/nemo-reranker-large
    reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        reranker_model = reranker_model.cuda()
    reranker_model.eval()


def rerank_documents(query: str, documents: List[str], top_k: int = 4) -> List[str]:
    """Rerank documents using NVIDIA's reranking model"""
    if reranker_model is None or reranker_tokenizer is None:
        initialize_reranker()

    pairs = [[query, doc] for doc in documents]
    features = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    if torch.cuda.is_available():
        features = {k: v.cuda() for k, v in features.items()}

    with torch.no_grad():
        scores = reranker_model(**features).logits.squeeze(-1)
        scores = scores.cpu().numpy()

    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in ranked_indices]


class HybridRetriever(BaseRetriever, BaseModel):
    """Custom hybrid retriever combining BM25 and Vector search with reranking"""
    vectorstore: FAISS = Field(description="FAISS vector store")
    bm25_retriever: BM25Retriever = Field(description="BM25 retriever")
    k: int = Field(default=4, description="Number of documents to retrieve")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid search and reranking"""
        # Get documents from both retrievers
        vector_docs = self.vectorstore.similarity_search(query, k=self.k)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # Combine unique documents
        all_docs = []
        seen_content = set()

        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc.page_content)
                seen_content.add(doc.page_content)

        # Rerank combined documents
        reranked_contents = rerank_documents(query, all_docs, self.k)

        # Convert back to Documents
        return [Document(page_content=content) for content in reranked_contents]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        raise NotImplementedError("Async retrieval not implemented")


def create_vector_database(output_path, data_path="/Users/luuhoanglinh/School/FinalProject/Project/kma_chat_agent/src/agent/regulation.txt"):
    """Create and save the vector database"""
    try:
        # Load the regulation document with UTF-8 encoding
        with open(data_path, "r", encoding="utf-8") as f:
            regulations = f.read()

        # Split the text into chunks with Vietnamese-aware settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Prioritize natural breaks
            keep_separator=False
        )
        chunks = text_splitter.split_text(regulations)

        # Create embeddings using Linq-Embed-Mistral
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)

        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            os.makedirs(output_path, exist_ok=True)

        vectorstore.save_local(output_path)
        return chunks
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        raise
    except Exception as e:
        print(f"Error creating vector database: {e}")
        raise


def load_vector_database(output_path):
    """Load the vector database"""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        if not os.path.exists(output_path):
            chunks = create_vector_database(output_path)
        else:
            # If vector database exists, we still need to get the chunks
            with open("/Users/luuhoanglinh/School/FinalProject/Project/kma_chat_agent/src/agent/regulation.txt", "r", encoding="utf-8") as f:
                regulations = f.read()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
                keep_separator=False
            )
            chunks = text_splitter.split_text(regulations)

        print("Loading vector database...")

        return FAISS.load_local(output_path, embeddings, allow_dangerous_deserialization=True), chunks
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        raise
    except Exception as e:
        print(f"Error loading vector database: {e}")
        raise


def get_prompt():
    """Get the chat prompt template"""
    prompt_template = """
    Bạn là một trợ lý ảo dành cho sinh viên và giảng viên của Học viện Kỹ thuật mật mã (viết tắt là KMA).
    Bạn có hiểu biết về tất cả các quy định và chính sách của trường và có thể giúp đỡ với bất kỳ câu hỏi nào về chúng.
    Bạn sẽ dựa trên thông tin từ tài liệu được cung cấp bên dưới để trả lời câu hỏi của người dùng.
    Hãy trả lời các câu hỏi dưới vai trò một trợ lý ảo thông minh và chuyên nghiệp. Trả lời chính xác, ngắn gọn.
    Nếu không thể trả lời, hãy nói rằng bạn không thể trả lời câu hỏi đó.
    Hãy trả lời các câu hỏi bằng tiếng Việt.
    Luôn trả lời dưới dạng một đoạn văn hoàn chỉnh, không trả lời theo dạng danh sách.
    Luôn trả lời dưới dạng văn bản có định dạng, sử dụng định dạng markdown nếu cần thiết.

    Tài liệu: {context}
    Lịch sử trò chuyện giữa bạn và người dùng: {chat_history}
    Câu hỏi: {question}
    Answer:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])


def initialize_rag():
    """Initialize the RAG system"""
    try:
        # Load the vector database and documents
        vectorstore, documents = load_vector_database("vector_db")

        # Initialize BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(
            texts=documents,
            k=6
        )

        # Create hybrid retriever with reranking
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            k=4
        )

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )

        # Initialize LLM
        llm = OllamaLLM(model="llama3.2")

        # Create conversation chain
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            verbose=True,
            output_key='answer',
            combine_docs_chain_kwargs={"prompt": get_prompt()},
            callback_manager=callback_manager
        )

        return conversation
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        raise


def main():
    """Main function to run the chat application"""
    print("Initializing KMA Chat Assistant...")
    try:
        conversation = initialize_rag()
        print("\nKMA Chat Assistant is ready! Type 'quit' to exit.")
        print("Ask me anything about KMA!\n")

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Check if user wants to quit
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Have a great day!")
                    break

                # Skip empty inputs
                if not user_input:
                    continue

                # Get AI response
                print("\nAssistant: ", end="")
                response = conversation.invoke({
                    "question": user_input
                })
                print(response["answer"])

            except UnicodeDecodeError as e:
                print(f"\nError with text encoding: {e}")
                print("Please try again with different text.")
            except Exception as e:
                print("\nError during conversation:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\nDetailed error traceback:")
                traceback.print_exc()
                print("\nYou can continue chatting or type 'quit' to exit.")

    except Exception as e:
        print("\nFatal error during initialization:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nDetailed error traceback:")
        traceback.print_exc()
        print("\nPlease check your environment setup and try again.")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print("\nUnexpected error:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nDetailed error traceback:")
        traceback.print_exc()
        sys.exit(1)