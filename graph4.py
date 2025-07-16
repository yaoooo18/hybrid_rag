import os
import codecs
import hashlib
import logging
import time
import json
from typing import List, Tuple, Dict, Any, Set

# --- 1. 新增导入 ---
from dotenv import load_dotenv

from thefuzz import fuzz
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
# import hanlp # hanlp在此脚本中未被使用，可以注释掉或删除
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from pydantic import SecretStr
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_neo4j import Neo4jGraph

# --- 2. 在所有代码逻辑开始前，加载环境变量 ---
load_dotenv()

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_rule_generation_prompt() -> ChatPromptTemplate:
    # ... (此函数及之后的所有函数定义保持不变) ...
    system_message = SystemMessagePromptTemplate.from_template(
        "你是一位精通中国古典文学的分析专家。你的任务是从给定的文本片段中，识别出所有实体（特别是人物、神仙、妖怪），并总结出它们的各种不同称呼、别名或头衔。"
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
        请仔细阅读以下文本，找出其中所有实体的不同名称。然后，将它们组织成一个JSON对象。
        在这个JSON对象中：
        - 每个键（key）应该是该实体最常用或最正式的“规范名称”。
        - 每个值（value）应该是一个包含其所有别名、非正式称呼或头衔的列表。
        输出必须是一个格式严格的JSON对象，不要包含任何额外的解释、注释或markdown标记。
        现在，请根据以下文本生成规则：
        ---
        文本内容: {input}
        """
    )
    return ChatPromptTemplate.from_messages([system_message, human_message])


def generate_normalization_rules_from_text(text: str, llm: ChatOpenAI, sample_size: int = 5000) -> Dict[str, List[str]]:
    logging.info("开始从文本中自动生成实体规范化规则...")
    rule_generation_prompt = create_rule_generation_prompt()
    rule_generation_chain = rule_generation_prompt | llm
    sample_text = text[:min(len(text), sample_size)]
    try:
        response = rule_generation_chain.invoke({"input": sample_text})
        content = response.content.strip()
        match = re.search(r'```json\s*([\s\S]+?)\s*```', content)
        json_str = match.group(1) if match else content
        rules = json.loads(json_str)
        logging.info(f"成功生成 {len(rules)} 条规范化规则。")
        return rules
    except Exception as e:
        logging.error(f"生成规范化规则时发生错误: {e}")
        return {}


class EntityNormalizer:
    def __init__(self, rules: Dict[str, List[str]], similarity_threshold: int = 90):
        self.rules = rules
        self.canonical_map = self._create_canonical_map()
        self.similarity_threshold = similarity_threshold
        logging.info(f"实体规范器初始化完成，加载了 {len(self.rules)} 条规则。")

    def _create_canonical_map(self) -> Dict[str, str]:
        canonical_map = {}
        for canonical, aliases in self.rules.items():
            for alias in aliases:
                canonical_map[alias] = canonical
        return canonical_map

    def normalize(self, entity_name: str) -> str:
        name = entity_name.strip().strip('<>')
        if name in self.canonical_map:
            return self.canonical_map[name]
        if name in self.rules:
            return name
        best_match = None
        highest_score = 0
        for canonical_name in self.rules.keys():
            score = fuzz.token_set_ratio(name, canonical_name)
            if score > highest_score:
                highest_score = score
                best_match = canonical_name
        if highest_score >= self.similarity_threshold:
            return best_match
        return name


# --- 3. 修改全局变量的定义，从环境变量中读取 ---

# 从环境变量获取Neo4j的配置
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 从环境变量获取千问的API Key
QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")

# 检查环境变量是否成功加载
if not all([NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, QIANWEN_API_KEY]):
    raise ValueError("部分或全部环境变量未能加载，请检查您的 .env 文件是否正确配置！")

# 使用环境变量初始化对象
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    refresh_schema=False
)

llm = ChatOpenAI(
    temperature=0,
    model="qwen-plus",
    api_key=SecretStr(QIANWEN_API_KEY),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/"
)


def read_txt_files(directory: str) -> List[Tuple[str, str]]:
    # ... (此函数及之后的所有函数定义保持不变) ...
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with codecs.open(file_path, 'r', encoding='utf-8') as file:
                    results.append((filename, file.read()))
                logging.info(f"成功读取文件: {filename}")
            except Exception as e:
                logging.error(f"读取文件 {filename} 失败: {str(e)}")
    return results


def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 > chunk_size:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = (current_chunk + "\n" + para) if current_chunk else para
    if current_chunk: chunks.append(current_chunk)
    return chunks


def create_document_and_chunk_nodes(graph: Neo4jGraph, file_name: str, file_path: str, chunks: List[str]) -> List[
    Dict[str, Any]]:
    graph.query("MERGE (d:`__Document__` {fileName: $file_name}) SET d.uri = $file_path",
                {"file_name": file_name, "file_path": file_path})
    chunk_data = [{"id": hashlib.sha1(content.encode()).hexdigest(), "content": content, "position": i + 1,
                   "file_name": file_name} for i, content in enumerate(chunks)]
    graph.query("""
    UNWIND $chunk_data AS data
    MERGE (c:`__Chunk__` {id: data.id}) SET c.text = data.content, c.position = data.position, c.fileName = data.file_name
    WITH data, c
    MATCH (d:`__Document__` {fileName: data.file_name}) MERGE (c)-[:PART_OF]->(d)
    """, {"chunk_data": chunk_data})
    graph.query("""
    MATCH (curr:`__Chunk__` {fileName: $file_name}), (next:`__Chunk__` {fileName: $file_name, position: curr.position + 1})
    MERGE (curr)-[:NEXT_CHUNK]->(next)
    """, {"file_name": file_name})
    return chunk_data


def create_knowledge_extraction_prompt(entity_types: List[str], relation_types: List[str]) -> ChatPromptTemplate:
    system_message = SystemMessagePromptTemplate.from_template(
        "你是一个知识提取专家，从文本中提取实体和关系。"
        "你必须严格遵守指定的实体和关系类型。"
        "至关重要的一点是：如果文本中出现同一个实体的不同名称或别名（例如 '悟空', '齐天大圣'），"
        "你必须将它们统一识别为最常见或最完整的规范名称（例如 '孙悟空'）。"
        "输出必须是纯文本格式，不要包含任何解释或代码块。"
    )
    human_message = HumanMessagePromptTemplate.from_template(
        f"""
        请根据以下纲要从文本中提取知识。
        实体类型: {", ".join(entity_types)}
        关系类型: {", ".join(relation_types)}
        输出格式:
        [实体] <类型> <规范化名称> : <描述>
        [关系] <源实体规范名> -> <关系类型> -> <目标实体规范名> : <关系描述>
        现在，请处理以下文本内容：
        文本内容：{{input}}
        """
    )
    return ChatPromptTemplate.from_messages([system_message, human_message])


def parse_knowledge_output(output_text: str, normalizer: EntityNormalizer) -> GraphDocument:
    nodes, relationships, entity_map = [], [], {}
    for line in output_text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        try:
            if line.startswith('[实体]'):
                parts = re.split(r'[:：]', line[4:].strip(), maxsplit=1)
                if len(parts) < 2: continue
                entity_info = parts[0].strip().split(maxsplit=1)
                if len(entity_info) < 2: continue
                entity_type, entity_name_raw = entity_info[0].strip(), entity_info[1].strip()
                entity_name = normalizer.normalize(entity_name_raw)
                node = Node(id=entity_name, type=entity_type,
                            properties={"name": entity_name, "description": parts[1].strip()})
                nodes.append(node)
                entity_map[entity_name_raw.strip('<>')] = (entity_name, entity_type)
                entity_map[entity_name] = (entity_name, entity_type)
            elif line.startswith('[关系]'):
                parts = re.split(r'[:：]', line[4:].strip(), maxsplit=1)
                if len(parts) < 1: continue
                relation_parts = re.split(r'\s*->\s*', parts[0].strip())
                if len(relation_parts) != 3: continue
                source_raw, rel_type, target_raw = relation_parts[0].strip(), relation_parts[1].strip(), relation_parts[
                    2].strip()
                source_norm, source_type = entity_map.get(source_raw.strip('<>'),
                                                          (normalizer.normalize(source_raw), "实体"))
                target_norm, target_type = entity_map.get(target_raw.strip('<>'),
                                                          (normalizer.normalize(target_raw), "实体"))
                relationships.append(Relationship(source=Node(id=source_norm, type=source_type),
                                                  target=Node(id=target_norm, type=target_type), type=rel_type,
                                                  properties={
                                                      "description": parts[1].strip() if len(parts) > 1 else ""}))
        except Exception as e:
            logging.warning(f"解析行失败: '{line}' - {str(e)}")
    return GraphDocument(nodes=nodes, relationships=relationships, source=Document(page_content=""))


def store_knowledge(graph: Neo4jGraph, graph_doc: GraphDocument):
    """
    存储节点和关系，确保标签干净且属性被追加而不是覆盖。
    """
    # 1. 存储节点：不再使用:Entity硬编码标签，并追加描述
    node_creation_query = """
    UNWIND $nodes AS node_data
    // MERGE节点时只使用id属性，不指定标签
    MERGE (n {id: node_data.id})
    // 第一次创建时，设置名称和描述
    ON CREATE SET 
        n.name = node_data.id, 
        n.description = node_data.properties.description
    // 每次匹配到时，追加新的描述（如果新描述不为空）
    ON MATCH SET 
        n.description = CASE 
            WHEN node_data.properties.description <> '' THEN coalesce(n.description, '') + '\n' + node_data.properties.description 
            ELSE n.description 
        END
    // 使用APOC动态设置正确的标签（如：人物, 妖怪）
    WITH n, node_data
    CALL apoc.create.setLabels(n, [node_data.type]) YIELD node
    RETURN count(node)
    """
    nodes_as_dicts = [n.__dict__ for n in graph_doc.nodes]
    graph.query(node_creation_query, {"nodes": nodes_as_dicts})

    # 2. 存储关系：使用apoc.merge.relationship来避免重复，并追加描述
    relationship_creation_query = """
    UNWIND $rels AS rel_data
    // 匹配源节点和目标节点
    MATCH (source {id: rel_data.source_id})
    MATCH (target {id: rel_data.target_id})
    // 使用apoc.merge.relationship合并关系
    CALL apoc.merge.relationship(
        source, 
        rel_data.type, 
        {}, // onCreate属性
        {description: rel_data.description}, // onMatch属性（第一次也会执行）
        target
    ) YIELD rel
    // 在匹配时追加描述
    SET rel.description = CASE 
        WHEN rel.description IS NOT NULL AND rel.description <> rel_data.description 
        THEN rel.description + '\n' + rel_data.description 
        ELSE rel_data.description 
    END
    RETURN count(rel)
    """
    rels_as_dicts = []
    for rel in graph_doc.relationships:
        rels_as_dicts.append({
            "source_id": rel.source.id,
            "target_id": rel.target.id,
            "type": rel.type,
            "description": rel.properties.get("description", "")
        })

    if rels_as_dicts:
        graph.query(relationship_creation_query, {"rels": rels_as_dicts})


def extract_knowledge(chunk_content: str, knowledge_extraction_chain, normalizer: EntityNormalizer,
                      retries: int = 3) -> GraphDocument:
    for attempt in range(retries):
        try:
            response = knowledge_extraction_chain.invoke({"input": chunk_content})
            # --- 在这里添加打印语句 ---
            print("\n--- LLM Raw Output ---")
            print(response.content)
            print("--- End of Raw Output ---\n")
            # --- 添加结束 ---
            if not response.content.strip():
                logging.warning(f"知识提取尝试 {attempt + 1}/{retries}: LLM返回空内容。")
                time.sleep(2)
                continue
            graph_doc = parse_knowledge_output(response.content, normalizer)
            if graph_doc.nodes:
                logging.info(f"成功提取到 {len(graph_doc.nodes)} 个实体和 {len(graph_doc.relationships)} 个关系。")
                return graph_doc
        except Exception as e:
            logging.warning(f"知识提取失败 (尝试 {attempt + 1}/{retries}): {str(e)}")
            time.sleep(2)
    logging.warning("所有重试均失败，未能提取任何知识。")
    return GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))


def merge_duplicate_entities(graph: Neo4jGraph, normalizer: EntityNormalizer):
    """
    在数据库层面合并别名实体到规范实体，并智能合并它们的'description'属性。
    """
    logging.info("开始执行数据库层面的实体合并后处理...")
    try:
        graph.query("RETURN apoc.version()")
    except Exception:
        logging.warning("APOC插件未检测到。合并查询可能失败或效率低下。")
        return

    merge_count = 0
    for canonical_name, aliases in normalizer.rules.items():
        if not aliases: continue

        # 这个查询会找到规范节点和所有别名节点，
        # 将它们所有不为空的description收集起来，用换行符连接，
        # 然后执行合并，最后将合并好的description设置到最终的节点上。
        query = """
        // 1. 匹配规范节点
        MATCH (canonical {name: $canonical_name})
        // 2. 匹配所有别名节点
        UNWIND $aliases AS alias_name
        MATCH (alias {name: alias_name})
        // 确保它们不是同一个节点
        WHERE elementId(canonical) <> elementId(alias)

        // 3. 将规范节点和别名节点收集起来
        WITH canonical, COLLECT(alias) AS alias_nodes
        WHERE size(alias_nodes) > 0

        // 4. 收集所有节点（规范+别名）的描述，过滤掉空值
        WITH canonical, alias_nodes, [node IN alias_nodes + canonical WHERE node.description IS NOT NULL AND trim(node.description) <> ''] AS nodes_with_desc

        // 5. 将所有描述文本提取到一个列表中
        WITH canonical, alias_nodes, [n IN nodes_with_desc | n.description] AS descriptions

        // 6. 使用APOC函数将描述列表合并成一个由换行符分隔的字符串
        WITH canonical, alias_nodes, apoc.text.join(descriptions, '\n---\n') AS full_description

        // 7. 执行节点合并，这个操作会保留关系
        CALL apoc.refactor.mergeNodes(alias_nodes + [canonical], {mergeRels: true}) YIELD node

        // 8. 在合并后的最终节点上，设置完整的描述和正确的名称
        SET node.description = full_description, node.name = $canonical_name

        RETURN size(alias_nodes) AS merged_count
        """
        try:
            result = graph.query(query, {"canonical_name": canonical_name, "aliases": aliases})
            if result and result[0]:
                count = result[0].get('merged_count', 0)
                if count > 0:
                    logging.info(f"成功将 {count} 个别名节点合并到 '{canonical_name}'，并聚合了描述。")
                    merge_count += count
        except Exception as e:
            if "No data returned" in str(e):
                logging.info(f"对于 '{canonical_name}'，没有发现需要合并的别名节点。")
            else:
                logging.error(f"合并实体 '{canonical_name}' 时出错: {e}")
    logging.info(f"实体合并完成，总共合并了 {merge_count} 个节点。")


def create_schema_generation_prompt() -> ChatPromptTemplate:
    """
    创建一个专门用于生成Schema（实体和关系类型）的提示。
    """
    system_message = SystemMessagePromptTemplate.from_template(
        "你是一位资深的知识图谱架构师。你的任务是分析给定的文本，并为构建知识图谱提出一个合适的纲要（Schema）。"
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
        请仔细阅读以下文本，并根据其内容，推荐一个实体类型列表和一个关系类型列表。
        - 实体类型应该是名词性的，代表文本中的核心对象，例如：“人物”、“法宝”、“地点”。
        - 关系类型应该是动词性的，描述实体之间的联系，例如：“拥有”、“师徒”、“位于”。

        请将你的推荐结果组织成一个严格的JSON对象，格式如下：
        {{
          "entity_types": ["类型1", "类型2", ...],
          "relation_types": ["类型A", "类型B", ...]
        }}

        输出必须是纯粹的JSON对象，不要包含任何额外的解释、注释或markdown标记。
        现在，请根据以下文本生成纲要：
        ---
        文本内容: {input}
        """
    )
    return ChatPromptTemplate.from_messages([system_message, human_message])


def generate_schema_from_text(text: str, llm: ChatOpenAI, sample_size: int = 5000) -> Tuple[List[str], List[str]]:
    """
    使用LLM从文本中推荐实体和关系类型。
    """
    logging.info("开始从文本中自动推荐Schema（实体和关系类型）...")
    schema_prompt = create_schema_generation_prompt()
    schema_chain = schema_prompt | llm
    sample_text = text[:min(len(text), sample_size)]

    try:
        response = schema_chain.invoke({"input": sample_text})
        content = response.content.strip()
        match = re.search(r'```json\s*([\s\S]+?)\s*```', content)
        json_str = match.group(1) if match else content
        schema = json.loads(json_str)

        entities = schema.get("entity_types", [])
        relations = schema.get("relation_types", [])

        if not entities or not relations:
            logging.warning("LLM未能成功推荐有效的Schema，将返回空列表。")
            return [], []

        logging.info(f"成功推荐了 {len(entities)} 个实体类型和 {len(relations)} 个关系类型。")
        return entities, relations
    except Exception as e:
        logging.error(f"生成Schema时发生错误: {e}")
        return [], []


def _display_schema(entities: List[str], relations: List[str]):
    """辅助函数：清屏并显示当前的Schema。"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- Schema 审核与修改 ---")
    print("\n当前实体类型 (Entity Types):")
    if entities:
        for i, etype in enumerate(entities):
            print(f"  {i + 1}: {etype}")
    else:
        print("  (空)")

    print("\n当前关系类型 (Relation Types):")
    if relations:
        for i, rtype in enumerate(relations):
            print(f"  {i + 1}: {rtype}")
    else:
        print("  (空)")
    print("\n" + "=" * 40)


def _edit_schema_submenu(type_list: List[str], type_name: str) -> List[str]:
    """
    辅助函数：提供针对特定类型列表的编辑子菜单。
    支持增、删、改（全部批量）以及重构。
    """
    local_list = list(type_list)
    while True:
        # 为了清晰，只显示正在编辑的列表
        if type_name == "实体":
            _display_schema(local_list, [])
        else:
            _display_schema([], local_list)

        print(f"--- 正在编辑 {type_name} 类型 ---")
        print(f"  1. 增加 (批量)")
        print(f"  2. 删除 (批量)")
        print(f"  3. 修改 (批量)")
        print(f"  4. 重构 (用全新列表覆盖当前列表)")
        print("  5. 返回主菜单")

        choice = input("\n请输入选项 (1-5): ").strip()

        try:
            if choice == '1':  # 批量增加
                new_names_str = input(f"请输入要增加的 {type_name} 类型 (可输入多个，用空格或逗号分隔): ").strip()
                if not new_names_str: continue

                candidates = new_names_str.replace(',', ' ').split()
                added, skipped = [], []
                for name in candidates:
                    clean_name = name.strip()
                    if clean_name and clean_name not in local_list:
                        local_list.append(clean_name)
                        added.append(clean_name)
                    else:
                        skipped.append(clean_name)
                if added: print(f"成功增加: {', '.join(added)}")
                if skipped: print(f"已存在或无效，已跳过: {', '.join(skipped)}")
                time.sleep(2)

            elif choice == '2':  # 批量删除
                if not local_list: print("列表为空，无法删除。"); time.sleep(1); continue
                indices_str = input("请输入要删除的编号 (可输入多个，用空格或逗号分隔): ").strip()
                indices_to_delete = {int(item) - 1 for item in indices_str.replace(',', ' ').split() if
                                     item.isdigit()}
                valid_indices = sorted([i for i in indices_to_delete if 0 <= i < len(local_list)], reverse=True)
                if not valid_indices: print("未输入有效编号。"); time.sleep(1); continue

                deleted_items = [local_list.pop(idx) for idx in valid_indices]
                print(f"成功删除: {', '.join(reversed(deleted_items))}")
                time.sleep(2)

            elif choice == '3':  # 批量修改 (循环模式)
                if not local_list: print("列表为空，无法修改。"); time.sleep(1); continue
                print("进入批量修改模式。逐个输入要修改的编号，输入 'done' 结束。")
                while True:
                    if type_name == "实体":
                        _display_schema(local_list, [])
                    else:
                        _display_schema([], local_list)
                    print("--- 批量修改中 ---")

                    idx_str = input("请输入要修改的编号 (或输入 'done' 结束): ").strip().lower()
                    if idx_str == 'done': break

                    if not idx_str.isdigit() or not (0 <= int(idx_str) - 1 < len(local_list)):
                        print("编号无效，请重新输入。");
                        time.sleep(1);
                        continue

                    idx = int(idx_str) - 1
                    old_name = local_list[idx]
                    new_name = input(f"  -> 请输入 '{old_name}' 的新名称: ").strip()
                    if new_name and new_name not in local_list:
                        local_list[idx] = new_name
                        print(f"     成功将 '{old_name}' 修改为 '{new_name}'。")
                    else:
                        print("     新名称为空或已存在，修改失败。")
                    time.sleep(1)
                print("批量修改结束。")
                time.sleep(1)

            elif choice == '4':  # 重构当前列表
                print(f"\n--- {type_name} 列表重构 ---")
                print(f"当前的 {type_name} 列表将被完全覆盖。")
                new_items_str = input(f"请输入所有新的【{type_name}】类型 (用空格或逗号分隔): ").strip()

                # 使用与批量增加相同的逻辑来创建新列表，确保条目干净
                local_list = [item.strip() for item in new_items_str.replace(',', ' ').split() if item.strip()]

                print(f"{type_name} 列表已成功重构。")
                time.sleep(1.5)

            elif choice == '5':  # 返回
                return local_list
            else:
                print("无效选项，请输入1-5之间的数字。")
                time.sleep(1)

        except (ValueError, IndexError) as e:
            print(f"发生错误: {e}。请重试。")
            time.sleep(2)


def get_user_schema_confirmation(proposed_entities: List[str], proposed_relations: List[str]) -> Tuple[
    List[str], List[str]]:
    """
    向用户展示推荐的Schema，并提供一个功能强大的菜单驱动界面进行审核和修改。
    """
    entities = list(proposed_entities)
    relations = list(proposed_relations)

    while True:
        _display_schema(entities, relations)
        print("--- 主菜单 ---")
        print("  1. 编辑实体类型")
        print("  2. 编辑关系类型")
        print("  3. 完成并确认 Schema")
        print("  4. 放弃并退出程序")

        main_choice = input("\n请输入您的选择 (1-4): ").strip()

        if main_choice == '1':
            entities = _edit_schema_submenu(entities, "实体")
        elif main_choice == '2':
            relations = _edit_schema_submenu(relations, "关系")
        elif main_choice == '3':
            if not entities or not relations:
                print("\n错误：实体类型和关系类型列表都不能为空！")
                time.sleep(2)
                continue
            print("\nSchema已最终确认！继续执行后续流程...")
            return entities, relations
        elif main_choice == '4':
            print("用户选择退出程序。")
            exit()
        else:
            print("无效选择，请输入1-4之间的数字。")
            time.sleep(1)


def main():
    directory_path = r'C:\Users\lhy\Desktop\graph'
    file_contents = read_txt_files(directory_path)
    if not file_contents:
        logging.info("未找到任何文本文件，程序退出。")
        return

    # 清空数据库以便观察干净的结果
    logging.info("正在清空数据库...")
    graph.query("MATCH (n) DETACH DELETE n")

    combined_text = "\n\n".join([content for _, content in file_contents])

    # --- 新增工作流 ---
    # 1. LLM推荐Schema
    proposed_entities, proposed_relations = generate_schema_from_text(combined_text, llm)
    if not proposed_entities or not proposed_relations:
        logging.error("无法从文本中生成初始Schema，程序退出。")
        return

    # 2. 用户审核和确认Schema
    final_entities, final_relations = get_user_schema_confirmation(proposed_entities, proposed_relations)
    # --- 新增工作流结束 ---

    # 使用用户确认后的Schema进行后续操作
    logging.info(f"最终确认的实体类型: {final_entities}")
    logging.info(f"最终确认的关系类型: {final_relations}")

    generated_rules = generate_normalization_rules_from_text(combined_text, llm)
    normalizer = EntityNormalizer(rules=generated_rules, similarity_threshold=90)

    # 使用最终确认的类型列表来创建提取提示
    knowledge_extraction_prompt = create_knowledge_extraction_prompt(final_entities, final_relations)
    knowledge_extraction_chain = knowledge_extraction_prompt | llm

    for file_name, content in file_contents:
        logging.info(f"开始处理文件: {file_name}")
        # 注意：这里的chunk_size=100非常小，可能会影响提取效果，建议根据文本调整
        chunks = chunk_text(content, chunk_size=1000)  # 示例：将分块大小调整为1000
        logging.info(f"文件 {file_name} 分成 {len(chunks)} 个块")

        # 如果您不希望在图谱中看到 __Document__ 和 __Chunk__ 节点，可以注释掉下面这行
        chunk_data = create_document_and_chunk_nodes(graph, file_name, os.path.join(directory_path, file_name), chunks)

        for i, chunk in enumerate(chunk_data):
            logging.info(f"处理分块 {i + 1}/{len(chunks)}...")
            try:
                graph_doc = extract_knowledge(chunk_content=chunk['content'],
                                              knowledge_extraction_chain=knowledge_extraction_chain,
                                              normalizer=normalizer)
                if not graph_doc.nodes:
                    continue
                store_knowledge(graph, graph_doc)
                logging.info(f"成功存储分块 {i + 1} 的知识。")
            except Exception as e:
                logging.error(f"处理或存储分块 {i + 1} 时发生严重错误: {str(e)}", exc_info=True)

    merge_duplicate_entities(graph, normalizer)
    logging.info("知识图谱构建流程全部完成！")


if __name__ == "__main__":
    main()