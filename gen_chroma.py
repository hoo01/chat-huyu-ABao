from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import os

#指定两个文本文件的路径
file_path_1='./data/baozong_dialogues.txt'
file_path_2='./data/baozong_taici.txt'

#创建文本分割器实例
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

embedding_function = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

persist_directory ='./chroma' #路径如果报错就修改成绝对路径

#读取和分割文本文件
def split_text_file(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    for doc in docs:
        doc.metadata['custom_topic'] = 'abao'  # 添加topic字段，后来发现是chroma版本问题，不创建应该也行
    docs = text_splitter.split_documents(docs)
    return docs

def generate_split_docs():
    split_docs_1 = split_text_file(file_path_1)
    split_docs_2 = split_text_file(file_path_2)
    split_docs = split_docs_1 + split_docs_2
    return split_docs

# 当直接运行脚本时执行的代码
if __name__ == "__main__":
    split_docs = generate_split_docs()
    fanhua_vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_name='fanhua_collection'
    )
    fanhua_vectordb.persist()
    print("文本分割和向量数据库构建完成，并已持久化到磁盘。")

# 导出split_docs变量
__all__ = ['generate_split_docs']
