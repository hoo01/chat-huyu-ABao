import streamlit as st
from ragchat import Model_center

model_center = Model_center()
if "messages" not in st.session_state:
    st.session_state["messages"] = []     
# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("chat-沪语-阿宝")
    "[InternLM](https://github.com/InternLM/InternLM.git)"
    "[chat-huyu-ABao](https://github.com/hoo01/chat-huyu-ABao.git)"

# 创建一个标题和一个副标题
st.title("🌷🤵chat-沪语-阿宝")
st.caption("🚀 A streamlit chatbot powered by InternLM2 QLora")
    
# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["assistant"])

# Get user input
if prompt := st.chat_input("提出一个问题"):
    # Display user input
    st.chat_message("user").write(prompt)
        # 使用 qa_chain 生成回答
    response = model_center.qa_chain_self_answer(prompt)
    
    # 将问答结果添加到 session_state 的消息历史中
    st.session_state.messages.append({"user": prompt, "assistant": response}) 
    # 显示回答
    st.chat_message("assistant").write(response)