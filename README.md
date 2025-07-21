 # hybridRAG: 知识图谱与向量数据库混合检索系统

![alt text](https://img.shields.io/badge/python-3.9%2B-blue)
![alt text](https://img.shields.io/badge/license-MIT-green)
![alt text](https://img.shields.io/badge/status-active-brightgreen)

本项目实现了一个先进的检索增强生成（RAG）系统，它巧妙地结合了**知识图谱（Knowledge Graph）和向量数据库（Vector Database）**的优势，以提供更精确、更具上下文感知能力的检索结果。

知识图谱 (Neo4j): 用于存储结构化的、事实性的信息（如实体、属性、关系）。它能提供精确、可解释的答案。

向量数据库 (ChromaDB): 用于存储非结构化文本的语义嵌入。它擅长理解自然语言的模糊性和上下文，能召回语义上相似的内容。

通过结合两者，本系统能够克服单一检索方法的局限性，为下游的大语言模型（LLM）提供更丰富、更准确的上下文信息。

# ✨ 主要特性

多格式文件摄入: 自动读取并解析 .txt, .pdf, 和 .docx 文件。

自动化知识图谱构建: 使用大语言模型（LLM）从文本中自动提取实体和关系，并存入Neo4j。

交互式Schema定义: 允许用户在图谱构建前，审核、修改由LLM推荐或预定义的实体与关系类型。

智能文本分块: 采用 spaCy 的多语言模型进行句子级分割，确保在处理中英文混合文本时，最大程度地保留语义完整性。

增量式向量数据库: vector.py 支持增量更新，只会处理新添加到目录中的文件，避免重复工作。

混合检索: hybrid_search.py 结合了来自知识图谱的精确结果和向量数据库的语义相关结果。

# 🚀 架构概览
Generated mermaid
graph TD

    A[源文件 .txt, .pdf, .docx] --> B(graph.py);
    A --> C(vector.py);

    B -- 提取实体与关系 --> D[Neo4j 知识图谱];
    C -- 文本分块与嵌入 --> E[ChromaDB 向量数据库];

    F[用户查询] --> G(hybrid_search.py);
    D -- 结构化事实检索 --> G;
    E -- 语义相似度检索 --> G;

    G -- 融合与排序 --> H[混合检索结果];
    H --> I[大语言模型 LLM];
    I --> J[最终答案];

# 🛠️ 步骤一：环境配置与安装

在运行任何代码之前，请务必完成以下所有配置步骤。

1. Neo4j 数据库安装与配置

本项目使用 Neo4j 作为知识图谱的存储后端。

下载 Neo4j Desktop: 前往 Neo4j 官方下载页面 下载并安装适用于您操作系统的 Neo4j Desktop。

创建数据库:

打开 Neo4j Desktop，创建一个新的本地项目。

在项目中，点击 Add -> Local DBMS 创建一个新的数据库。

为数据库设置一个密码（例如 password），并牢记这个密码。

点击 Start 启动数据库。

安装 APOC 插件 (关键步骤):

选中您刚刚创建并已启动的数据库。

在右侧的详情面板中，点击 Plugins 标签页。

在列表中找到 APOC 插件，点击其右侧的 Install 按钮进行安装。

安装完成后，需要重启数据库才能使插件生效。

2. Python 环境

建议使用 conda 创建虚拟环境以避免包版本冲突。

在 Anaconda Prompt 中运行：

```Generated bash
conda env create -f environment.yaml
```
> **注意**:  `environment.yaml` 文件已经给出
如果不行，可以尝试将yaml文件中,pip：以下的文字全部复制到requirement.txt
 ```Generated bash
pip install -r requirements.txt
```
### 3. 配置环境变量

这是整个项目中最重要的一步。

1.  在项目根目录下，创建一个名为 `.env` 的文件。
2.  打开 `.env` 文件，参照 `环境.txt`（或以下示例）的格式，填入您的个人API密钥和数据库凭据。

**`.env` 文件内容示例:**
```env
 # Neo4j 数据库配置
 # 请确保这里的密码与您在 Neo4j Desktop 中设置的密码一致
 NEO4J_URL="bolt://localhost:7687"
 NEO4J_USER="neo4j"
 NEO4J_PASSWORD="your_neo4j_password_here"

 # 大语言模型 API Key (用于知识提取)
 # 例如：阿里云Dashscope的千问模型
 QIANWEN_API_KEY="your_qianwen_api_key_here"

 # 向量嵌入模型 API Key (用于向量数据库)
 # 例如：智谱AI的Embedding模型
 ZHIPUAI_API_KEY="your_zhipu_api_key_here"
```
# ⚙️ 步骤二：执行流程

请严格按照以下顺序执行脚本。

提示: 请始终运行文件名中数字最大的那个版本，因为它代表了最新的功能。

1. 放置源文件

将您所有需要处理的 .txt, .pdf, .docx 文件放入项目根目录下的 graph 文件夹（如果文件夹不存在，请手动创建）。

2. 构建知识图谱

运行 graph 脚本。这个脚本会读取源文件，让您确认Schema，然后提取实体和关系，最终构建知识图谱。

Generated bash
```
# 假设 graph6.py 是最新版本
python graph6.py
```
执行完毕后，您可以在 Neo4j Browser 中通过运行以下 Cypher 查询来查看生成的图谱：

Generated cypher
```
MATCH (n) RETURN n```
```
### 3. 构建向量数据库

运行 `vector` 脚本。这个脚本会读取相同的文件，使用 `spaCy` 进行智能分块，然后将文本块嵌入并存入 ChromaDB 向量数据库。

```bash
# 假设 vector3.py 是最新版本
python vector3.py
```
```> **增量更新**: 此脚本支持增量更新。如果您第一次运行后，又向 `graph` 文件夹添加了新文件，再次运行此脚本将只处理新添加的文件。

### 4. 进行混合检索

一切准备就绪后，运行 `hybrid` 脚本来体验混合检索的效果。

```bash
# 假设 hybrid_search1.py 是最新版本
python hybrid_search1.py
```
# 📂 文件说明

graph<N>.py: 负责知识图谱的端到端构建流程。

vector<N>.py: 负责向量数据库的端到端构建与增量更新流程。

hybrid_search<N>.py: 负责实现混合检索的查询逻辑。

.env: 存储所有敏感的API密钥和数据库凭据。请务必将此文件添加到 .gitignore 中，不要上传到GitHub！
