import os
import re
import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from llm import InternLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from gen_chroma import generate_split_docs
from modelscope import snapshot_download
# 配置日志
log_filename = 'ragchat.log' 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
split_docs = generate_split_docs()

def load_chain(split_docs):
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    
    # 向量数据库持久化路径
    persist_directory = './chroma' #根据下载好的模型的路径调整，如果路径报错就写绝对路径

    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory)
    vectordb.persist()
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    
    model_id = 'hooo01/chat-huyu-ABao'
    model_name_or_path = snapshot_download(model_id, revision='master')
    
    # 加载自定义 LLM
    llm = InternLM(model_path=model_name_or_path)

    # 定义一个 Prompt Template
    template = """现在你要扮演阿宝：阿宝，是繁花中的人物，生活在上世纪80年代的上海。阿宝是读者的朋友，愿意分享见闻，解答读者关于《繁花》或更广泛话题的好奇。记住阿宝是上海人，用上海方言回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    **注意**：如果能找到上下文，务必使用知识库回答，找不到再使用模型本身的知识。
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain(split_docs)

    def qa_chain_self_answer(self, question):
        """
        调用问答链进行回答，如果没有找到相关文档，则使用模型自身的回答
        """
        if not question:
            return "题目勿好空落落哦，请提供个有用个问题。"

        try:
            # 使用检索链来获取相关文档
            result = self.chain.invoke({"query": question})         
            print(f"Debugging: Result structure => {result}")
            
            if 'result' in result:
                answer = result['result']
                final_answer = re.sub(r'^阿宝：\s?', '', answer, flags=re.M).strip()
                final_answer = re.sub(r'^问题：\s?', '', final_answer, flags=re.M).strip()
                final_answer = re.sub(r'^阿宝\s?', '', final_answer, flags=re.M).strip()
                return final_answer
            else:
                print("Error: 'result' field not found in the result.")
                return "阿宝目前无法提供答案，请稍后再试。"
        except Exception as e:
            # 打印更详细的错误信息，包括traceback
            import traceback
            print(f"An error occurred: {e}\n{traceback.format_exc()}")
            return "阿宝遇到了一些技术问题，正在修复中。"