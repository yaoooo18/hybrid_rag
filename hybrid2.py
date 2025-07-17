import os
import logging
from typing import List, Dict
import jieba
import jieba.posseg as pseg
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# 修复导入问题
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import ZhipuAIEmbeddings
    from langchain_community.graphs import Neo4jGraph
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_community.chat_models import ChatOpenAI

    logging.info("成功导入所有必要的库")
except ImportError as e:
    logging.error(f"导入失败: {str(e)}")
    logging.info("请安装必要的库: pip install langchain-community langchain-openai python-dotenv jieba")


class HybridRAGSystem:
    def __init__(self):
        # 加载向量数据库
        self.vectordb = self._load_vector_db()

        # 连接知识图谱
        self.graph = self._load_knowledge_graph()

        # 初始化LLM
        self.llm = self._load_llm()

        # 创建LLM链
        self.llm_chain = self._create_llm_chain()

        # 获取知识图谱属性配置
        self.entity_property = "name"  # 根据您的图谱使用 "name"

        # 获取知识图谱实体类型
        self.entity_types = self._get_entity_types()
        logging.info(f"知识图谱实体类型: {self.entity_types}")

        logging.info("混合RAG系统初始化完成")

    def _get_entity_types(self) -> List[str]:
        """从知识图谱中获取所有实体类型"""
        if not self.graph:
            return ["Entity"]  # 默认值

        try:
            # 查询所有节点标签
            query = """
            CALL db.labels() YIELD label
            WHERE label <> '__Chunk__' AND label <> '__Document__'
            RETURN label
            """
            result = self.graph.query(query)
            return [item["label"] for item in result if item["label"]]
        except Exception as e:
            logging.error(f"获取实体类型失败: {str(e)}")
            return ["Entity"]  # 默认值

    def _load_vector_db(self) -> Chroma:
        """加载Chroma向量数据库"""
        try:
            # 从环境变量获取API密钥
            zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")
            if not zhipu_api_key:
                logging.warning("未找到ZHIPUAI_API_KEY环境变量")

            # 创建嵌入函数
            embeddings = ZhipuAIEmbeddings(api_key=zhipu_api_key)

            # 加载向量数据库
            vectordb = Chroma(
                persist_directory="LLM1_ChromaDB_Spacy",
                embedding_function=embeddings
            )
            logging.info("向量数据库加载成功")
            return vectordb
        except Exception as e:
            logging.error(f"向量数据库加载失败: {str(e)}")
            return None

    def _load_knowledge_graph(self) -> Neo4jGraph:
        """连接Neo4j知识图谱"""
        try:
            # 从环境变量获取连接信息
            neo4j_url = os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "liuhaoyao")

            # 连接知识图谱
            graph = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password,
                refresh_schema=False
            )
            logging.info(f"知识图谱连接成功: {neo4j_url}")
            return graph
        except Exception as e:
            logging.error(f"知识图谱连接失败: {str(e)}")
            return None

    def _load_llm(self) -> ChatOpenAI:
        """初始化大语言模型"""
        try:
            # 从环境变量获取API密钥
            api_key = os.getenv("QIANWEN_API_KEY", "sk-993482af36124cc1b4c6e137ee8b6bdf")

            # 初始化LLM
            return ChatOpenAI(
                temperature=0.3,
                model="qwen-plus",
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/"
            )
        except Exception as e:
            logging.error(f"LLM初始化失败: {str(e)}")
            return None

    def _create_llm_chain(self) -> LLMChain:
        """创建LLM问答链"""
        prompt_template = """
        你是一个专业的知识助手，请基于以下混合检索结果回答用户问题：
        禁止使用任何外部知识或常识回答问题
        {context}

        ### 用户问题:
        {question}

        ### 回答要求:
        1. 优先使用向量数据库的文本内容
        2. 用知识图谱补充实体关系
        3. 如信息冲突，以向量数据库为准
        4. 如不确定，请说明信息来源
        5. 保持专业、简洁的回答风格
        6. 使用中文回答
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        if self.llm is None:
            logging.error("无法创建LLM链，LLM未初始化")
            return None

        return LLMChain(llm=self.llm, prompt=prompt)

    def retrieve_from_vector_db(self, query: str, k: int = 3) -> List[Dict]:
        """从向量数据库中检索相关文档"""
        if not self.vectordb:
            logging.warning("向量数据库未加载")
            return []

        try:
            results = self.vectordb.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "向量数据库"
                }
                for doc in results
            ]
        except Exception as e:
            logging.error(f"向量检索失败: {str(e)}")
            return []

    def retrieve_from_knowledge_graph(self, query: str) -> List[Dict]:
        """从知识图谱中检索相关信息"""
        if not self.graph:
            logging.warning("知识图谱未连接")
            return []

        try:
            # 提取实体
            entities = self.extract_entities(query)
            if not entities:
                logging.info(f"未提取到实体: '{query}'")
                return []

            # 查询图谱
            results = []
            for entity in entities:
                graph_results = self.query_graph(entity)
                if graph_results:
                    results.append({
                        "content": graph_results,
                        "entity": entity,
                        "source": "知识图谱"
                    })
            return results
        except Exception as e:
            logging.error(f"知识图谱检索失败: {str(e)}")
            return []

    def extract_entities(self, query: str) -> List[str]:
        """使用jieba分词提取中文实体"""
        try:
            # 使用jieba进行分词和词性标注
            words = pseg.cut(query)

            # 提取名词性短语作为实体
            entities = []
            for word, flag in words:
                # 提取名词（包括人名、地名、机构名等）
                if flag.startswith('n'):
                    entities.append(word)
                # 提取动词作为可能的实体
                elif flag.startswith('v') and len(word) > 1:
                    entities.append(word)

            # 如果提取的实体太少，尝试提取所有有意义的名词
            if len(entities) < 2:
                for word, flag in words:
                    if len(word) > 1 and flag not in ['x', 'm', 'q']:  # 排除标点、数字等
                        entities.append(word)

            # 去重并限制数量
            return list(set(entities))[:3]  # 返回最多3个唯一实体
        except Exception as e:
            logging.error(f"实体提取失败: {str(e)}")
            # 回退方法：使用整个查询作为实体
            return [query]

    def query_graph(self, entity: str) -> str:
        """查询知识图谱获取实体相关信息"""
        if not self.graph:
            return f"知识图谱未连接"

        if not self.entity_types:
            return f"没有可用的实体类型"

        try:
            # 收集所有关系
            all_relations = []

            # 对每个实体类型分别查询
            for entity_type in self.entity_types:
                # 安全构建查询 - 使用参数化查询避免注入
                query = f"""
                MATCH (e:`{entity_type}`)-[r]->(related)
                WHERE e.{self.entity_property} CONTAINS $entity
                RETURN e.{self.entity_property} AS entity, 
                       type(r) AS relation, 
                       related.{self.entity_property} AS related_entity
                LIMIT 3
                """

                logging.info(f"执行知识图谱查询: {query}")
                data = self.graph.query(query, params={"entity": entity})

                if data:
                    # 格式化结果
                    for item in data:
                        entity_name = item.get("entity", "未知实体")
                        relation_type = item.get("relation", "未知关系")
                        related_entity = item.get("related_entity", "未知实体")
                        all_relations.append(f"{entity_name} -[{relation_type}]-> {related_entity}")

            if not all_relations:
                return f"未找到实体 '{entity}' 的相关信息"

            # 返回前3个关系
            return "\n".join(all_relations[:3])
        except Exception as e:
            logging.error(f"图谱查询失败: {str(e)}")
            return f"查询实体 '{entity}' 时出错"

    def generate_context(self, vector_results: List[Dict], graph_results: List[Dict]) -> str:
        """生成混合上下文"""
        context_parts = []

        # 添加向量数据库结果
        if vector_results:
            context_parts.append("### 向量数据库检索结果:")
            for i, result in enumerate(vector_results, 1):
                source = result.get("metadata", {}).get("source", "未知来源")
                context_parts.append(f"{i}. [{source}] {result['content']}")

        # 添加知识图谱结果
        if graph_results:
            context_parts.append("\n### 知识图谱检索结果:")
            for result in graph_results:
                context_parts.append(f"- 实体: {result['entity']}")
                context_parts.append(f"  关系: {result['content']}")

        return "\n".join(context_parts) if context_parts else "未检索到相关信息"

    def ask(self, question: str) -> str:
        """混合RAG问答接口"""
        if self.llm_chain is None:
            logging.error("LLM链未初始化")
            return "系统初始化失败，无法回答问题"

        # 从向量数据库检索
        vector_results = self.retrieve_from_vector_db(question)

        # 从知识图谱检索
        graph_results = self.retrieve_from_knowledge_graph(question)

        # 生成混合上下文
        context = self.generate_context(vector_results, graph_results)

        # 生成回答 - 使用推荐的invoke方法替代run
        try:
            # 使用推荐的invoke方法
            response = self.llm_chain.invoke(
                input={"context": context, "question": question}
            )
            return response['text'].strip()
        except Exception as e:
            logging.error(f"生成回答失败: {str(e)}")
            return "抱歉，生成回答时出错"


# 交互式对话系统
def interactive_chat():
    # 初始化系统
    rag_system = HybridRAGSystem()

    # 显示欢迎信息
    print("=" * 70)
    print("欢迎使用混合RAG问答系统")
    print("输入您的问题，系统将结合向量数据库和知识图谱提供专业回答")
    print("输入 '退出'、'exit' 或 'q' 结束对话")
    print("=" * 70)

    # 对话计数器
    conversation_count = 0

    while True:
        # 获取用户输入
        user_input = input("\n您: ")

        # 检查退出命令
        if user_input.lower() in ['退出', 'exit', 'q']:
            print("\n感谢使用混合RAG问答系统，再见！")
            break

        # 空输入处理
        if not user_input.strip():
            print("请提出您的问题")
            continue

        # 增加对话计数
        conversation_count += 1

        # 获取回答
        print(f"\n系统思考中... (第 {conversation_count} 轮对话)")
        answer = rag_system.ask(user_input)

        # 输出回答
        print("\n助手:")
        print(answer)
        print("-" * 70)


# 主程序入口
if __name__ == "__main__":
    interactive_chat()
