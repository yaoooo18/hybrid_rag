import os
import codecs
import logging
from typing import List, Tuple, Dict, Any

# 使用新版本的导入方式
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RAGIngestor:
    """
    一个用于处理文档、生成嵌入并将其存储到ChromaDB的类。
    核心功能是支持增量更新，避免重复处理已存在的文件。
    """

    def __init__(self, persist_dir: str, embeddings_model: str = "embedding-2"):
        """
        初始化Ingestor。
        :param persist_dir: ChromaDB持久化存储的路径。
        :param embeddings_model: 使用的嵌入模型名称。
        """
        self.persist_dir = persist_dir
        self.embeddings = ZhipuAIEmbeddings(model=embeddings_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectordb = self._load_or_create_vectordb()

    def _load_or_create_vectordb(self) -> Chroma:
        """加载现有的ChromaDB，如果不存在则创建一个新的。"""
        if os.path.exists(self.persist_dir):
            logging.info(f"从 {self.persist_dir} 加载已存在的向量数据库...")
            return Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        else:
            logging.info("未找到现有数据库，将创建一个新的。")
            # 创建一个空的数据库，后续通过 .add_documents() 添加
            # 这里我们先返回None，在第一次添加时创建
            return None

    def _get_ingested_files(self) -> set:
        """从向量数据库中获取所有已经处理过的文件名。"""
        if not self.vectordb:
            return set()

        try:
            # get()方法可以获取所有数据，但可能非常耗时和耗内存
            # 对于大型数据库，建议使用更高效的元数据查询方式
            # 这里为了演示，我们获取所有元数据
            existing_docs = self.vectordb.get(include=["metadatas"])
            ingested_sources = {meta['source'] for meta in existing_docs['metadatas']}
            logging.info(f"数据库中已存在 {len(ingested_sources)} 个来源文件。")
            return ingested_sources
        except Exception as e:
            logging.error(f"从数据库获取元数据失败: {e}")
            return set()

    def _read_new_files(self, directory: str, ingested_files: set) -> List[Document]:
        """读取目录中尚未被处理的新文件。"""
        documents = []
        logging.info(f"开始扫描目录 {directory} 中的新文件...")
        for filename in os.listdir(directory):
            if filename.endswith(".txt") and filename not in ingested_files:
                file_path = os.path.join(directory, filename)
                try:
                    with codecs.open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    # 为文档添加丰富的元数据
                    metadata = {"source": filename}
                    documents.append(Document(page_content=content, metadata=metadata))
                    logging.info(f"发现并读取新文件: {filename}")
                except Exception as e:
                    logging.error(f"读取文件 {filename} 失败: {str(e)}")
        return documents

    def ingest_from_directory(self, directory: str):
        """
        从指定目录执行增量式的数据摄入。
        """
        # 1. 获取已处理的文件列表
        ingested_files = self._get_ingested_files()

        # 2. 读取新文件
        new_documents = self._read_new_files(directory, ingested_files)

        if not new_documents:
            logging.info("没有发现需要处理的新文件。")
            return

        # 3. 分割新文档
        logging.info(f"正在分割 {len(new_documents)} 个新文档...")
        split_docs = self.text_splitter.split_documents(new_documents)

        # 优化：为每个块添加更丰富的元数据
        for doc in split_docs:
            source = doc.metadata["source"]
            # 简单的块编号，可以根据需要设计更复杂的
            if not hasattr(self, f'chunk_count_{source}'):
                setattr(self, f'chunk_count_{source}', 0)

            chunk_num = getattr(self, f'chunk_count_{source}')
            doc.metadata['chunk_number'] = chunk_num
            setattr(self, f'chunk_count_{source}', chunk_num + 1)

        logging.info(f"新文档被分割成 {len(split_docs)} 个块。")

        # 4. 将新文档添加到向量数据库
        if self.vectordb is None:
            # 首次创建数据库
            logging.info("正在创建新的向量数据库并添加文档...")
            self.vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            # 向现有数据库添加文档
            logging.info("正在向现有数据库中添加新文档...")
            self.vectordb.add_documents(split_docs)

        # 5. 持久化
        logging.info("正在持久化数据库...")
        self.vectordb.persist()
        logging.info(f"数据摄入完成！向量数据库已更新并保存至: {os.path.abspath(self.persist_dir)}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 设置API密钥（建议在您的运行环境中设置，而不是在代码中）
    if "ZHIPUAI_API_KEY" not in os.environ:
        os.environ["ZHIPUAI_API_KEY"] = "8d49b3f1c7fb4fed8eeec073e49a8a9c.m432pWtUGn3ZEAtQ"

    # 2. 定义路径
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 向量数据库保存路径
    db_path = os.path.join(script_dir, "LLM1_ChromaDB")
    # 源文件目录
    source_directory = r'C:\Users\lhy\Desktop\graph'

    # 3. 初始化并运行Ingestor
    ingestor = RAGIngestor(persist_dir=db_path)
    ingestor.ingest_from_directory(directory=source_directory)

    print("\n--- RAG数据摄入流程执行完毕 ---")

    # (可选) 验证一下数据库内容
    if ingestor.vectordb:
        print(f"数据库中总文档块数: {ingestor.vectordb._collection.count()}")
        # 示例查询
        results = ingestor.vectordb.similarity_search("孙悟空有什么法宝？", k=2)
        print("\n示例查询结果:")
        for doc in results:
            print(f"来源: {doc.metadata.get('source')}, 块号: {doc.metadata.get('chunk_number')}")
            print(f"内容: {doc.page_content[:100]}...\n")