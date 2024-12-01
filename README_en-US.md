# 🌷🤵chat-HuYu-ABao

[![Static Badge](https://img.shields.io/badge/license-Apache%202.0-00a2a8)][license-url] | [![Static Badge](https://img.shields.io/badge/openxlab-models-blue)][OpenXLab_Model-url] | [![Static Badge](https://img.shields.io/badge/modelscope-models-9371ab)
][ModelScope-url]
[](./README_en.md)

[license-url]: ./LICENSE
[OpenXLab_Model-url]: https://openxlab.org.cn/models/detail/hoo01/chat-huyu-ABao
[ModelScope-url]: https://www.modelscope.cn/models/hooo01/chat-huyu-ABao

[English](./README_en-US.md) | [简体中文](./README.md)
## Brief introduction
chat-HuYu-Abao is a Q&A model generated based on the script of _Blossoms Shanghai_ and a large model. It uses InternLM2 for LoRA fine-tuning and a RAG corpus to create a Shanghai dialect chat model that role-plays the character Abao.

> Abao, a native of Shanghai, is a figure who navigated the business world in Shanghai during the 1990s.
> 
> Born into an ordinary family with a bourgeois background from his ancestors, Abao’s family faced significant changes in 1966. Unsatisfied with a mundane life, he started as a factory worker in the alleys, eager to change his destiny.
> 
> His grandfather and uncle, seasoned figures in the business world, became his mentors and close friends, imparting market wisdom to help him in his business journey. His first love, Xue Zhi, was a chance encounter on a bus. Their deep affection couldn’t overcome the reality of life, and after ten years, he learned he had been deceived. Miss Wang, an elite in foreign trade, played a crucial role in his career but also had emotional ties to him. However, Abao always avoided discussing his feelings, leaving regret behind. Ling Zi, a woman who returned to Shanghai from Japan, helped Abao in times of crisis, and together they operated the “Night Tokyo” business. Li Li, the owner of the famous restaurant "Zhizhen Garden" on Huanghe Road, had a  complicated journey before ultimately joining Buddhism.
> 
> Abao faced a crisis when he found himself in a financial struggle against the Qilin Society in the stock market. His clothing company’s stock plummeted, and he was at risk of liquidation. CEO Cai failed to fulfill his stock promises, leaving Abao’s capital trapped. Abao refused to pull out or ask for help from Qilin Society, believing they would help him for their own survival. Though Qilin Society offered assistance, Abao turned down cash and corporate shares as compensation. Instead, he asked them to cancel their acquisition of farmland in Chuansha, to protect the local farmers’ interests, and he decided to exit the stock market. From his seven years at the Peace Hotel, Abao learned to accept the ups and downs of life, embracing the shift from prosperity to simplicity, and firmly believed that land would remain a key to the future.

## System Architecture Diagram
![enter image description here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/%E8%8B%B1%E6%96%87%E6%9E%B6%E6%9E%84.png?raw=true)

## Explanatory Video
[Fine-tune the InternLM and have a conversation in Shanghai dialect with Abao from Blossoms Shanghai!](https://www.bilibili.com/video/BV1sJvFeXELe/?spm_id_from=333.999.0.0&vd_source=9b01f3d1e6addb97637b80b1bb9c008b)

## Project highlights

 1. Answer in Shanghai dialect. 
 2. Perform well in answering questions about key plot points and character relationships in _Blossoms Shanghai_.

## How to start

❕ Currently, only local deployment methods are provided. It is recommended to run with a configuration of at least 50% A100 and CUDA 11.7.

**1. Clone this project to your local development machine.** 

    git clone https://github.com/hoo01/chat-huyu-ABao.git

**2. Set up the environment.**

    # Create a virtual environment.
    conda create -n fanhua python=3.10 
    # Activate the environment.
    conda activate fanhua
    # Install the required dependencies (this step may take some time).
    cd chat-huyu-ABao  
    pip install -r requirements.txt

**3.Generate the Chroma database required for RAG dependencies.**

    python gen_chroma.py

**4.Start (This part may take a long time due to model download)**

    python start.py

> The default port and IP are 127.0.0.1:7860. To change them, please open start.py and modify the settings.<br>
> For non-local development machines, port mapping is required.<br>
>Taking the InternStudio development machine in Shanghai dialect as an example, open the command prompt on the local machine and enter the following:
> `ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p <your ID>`<br>
> password：input your password![enter image description
> here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/%E6%98%A0%E5%B0%841.png?raw=true)
> ![enter image description
> here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/%E6%98%A0%E5%B0%842.png?raw=true)

**5.Example output**
![enter image description here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/test1.jpg?raw=true)

## Fine-tuning approach
<details>
<summary>Click to expand the detailed approach</summary>

**1. Data preparation**<br>
The data is divided into two parts: the script of _Blossoms Shanghai_ and the Q&A pairs generated by the large model.<br>
1.Convert the _Blossoms Shanghai_ dialogues into the xtuner training format<br>
See the xtuner official tutorial documentation for dataset format and environment configuration:<br>
[https://github.com/InternLM/Tutorial/blob/camp2/data_fine_tuning/data_fine_tuning.md](https://github.com/InternLM/Tutorial/blob/camp2/data_fine_tuning/data_fine_tuning.md)
The format of the converted dialogues is as follows:

    `{"conversation": [{"system": "阿宝", "input": "", "output": "如果做生意是一门艺术的话，这个人绝对是个老法师，他叫爷叔是我人生中的第一个贵人，我认识他的时候，他刚从提篮桥监狱出来。"}, {"system": "阿宝", "input": "爷叔：谁啊。", "output": "爷叔好，我是阿宝。你不认识我了？"}, {"system": "阿宝", "input": "今天的太阳晒不到明天的衣裳的，时间，时间决定一切。回去吧。", "output": "爷叔我还没讲完呢。我是来跟你学生意的。"}, {"system": "阿宝", "input": "你知道什么叫生意？", "output": "这个不很简单的，生意嘛就是一买一卖，将本求利。"}, {"system": "阿宝", "input": "这是你爷爷告诉你的？你爷爷的爷爷是地主，到你爷爷这代是资本家，那么你呢。", "output": "我呢就是……"}, {"system": "阿宝", "input": "你是没钱开公司，那你原始积累怎么办，靠偷靠抢。", "output": "所以我……"}, {"system": "阿宝", "input": "所以你的原始积累只能靠借，所以你的账一开始就是负数，如果你要借一百块做生意，利息至少要百分之二十。你要做到多少利润，这生意可以继续下去。", "output": "一百二十块？"}, {"system": "阿宝", "input": "错，两百八十块。", "output": "为什么？"}, {"system": "阿宝", "input": "本金加利息一百二十块，生活二十块，成本四十块，剩下的一百块可以做生意了。一百块的成本，两百八十块的利润，什么概念？暴利，你知道我什么地方出来的。", "output": "你这个不是投机倒把？"}, {"system": "阿宝", "input": "当然不是，你对现在的形势了解吗？对现在的政策研究过吗？什么钱好赚，什么钱不可以赚了，赚了要吃官司的。拍拍胸脯，就要发财了，想也不要想。回去，回去。", "output": "爷叔，我是要做外贸的。"}, {"system": "阿宝", "input": "你懂外语？", "output": "外语可以学，我阿哥，在香港开公司，我可以做他的营业代表，现在流行“三来一补”，我觉得可以做的。"}, {"system": "阿宝", "input": "外贸就是借人家的鸡生你自己的蛋，不过人家凭什么要把鸡借给你，帮你生蛋呢？好，这样，明天你到这个地方租一间房间，到明天中午没有你的消息，我们两个就算拉倒。", "output": "和平饭店。"}]}`

2. Generate Q&A pairs using the API.<br>
2.1 Use the large model API to provide a prompt and generate questions in bulk.  
The complete script can be found in `data/数据准备/gen_q_api.ipynb`.
2.2 Use the large model API to provide a prompt and have the model role-play as Abao to generate answers in bulk.  
The complete script can be found in `data/数据准备/q2a_api.ipynb`.

3.  Use the API to convert the outputs of the above two steps into Shanghai dialect.  
    The complete script can be found in `data/数据准备/pth2huyu.py`.

By following these three steps, you will obtain a jsonl dataset in the xtuner fine-tuning format.

**2. Fine-tune the model.**<br>
Official tutorial for the xtuner fine-tuning  <br>
https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md
https://github.com/InternLM/Tutorial/blob/camp2/data_fine_tuning/data_fine_tuning.md<br>
1.Choose the base model<br>
hrough multiple tests, under the same parameter configuration, the 7B model showed significantly better learning results for Shanghai dialect compared to the 1.8B model. Therefore, the base model selected is internlm2-chat-7b<br>
2.Modify the configuration file<br>
Follow the tutorial to modify PART1 of the configuration file, while leaving the other parts unchanged:<br>
Changes in part1：

     # Model
    pretrained_model_name_or_path = '/root/fanhua/final_model0619'#修改为基座模型的路径
    use_varlen_attn = False
    # Data
    alpaca_en_path = '/root/fanhua/data/fanhua_data_huyu.jsonl'#修改原始数据集路径
    prompt_template = PROMPT_TEMPLATE.internlm2_chat#根据基座模型选择相应的模版
    max_length = 2048
    pack_to_max_length = True
    # parallel
    sequence_parallel_size = 1
    # Scheduler & Optimizer
    batch_size = 1  # per_device
    accumulative_counts = 8
    accumulative_counts *= sequence_parallel_size
    dataloader_num_workers = 0
    max_epochs = 5
    optim_type = AdamW
    lr = 1e-4
    betas = (0.9, 0.999)
    weight_decay = 0
    max_norm = 1  # grad clip
    warmup_ratio = 0.03
    # Save
    save_steps = 100
    save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)
    # Evaluate the generation performance during the training
    evaluation_freq = 200
    SYSTEM = SYSTEM_TEMPLATE.alpaca
    evaluation_inputs = [
    '"从一个普通青年到上海滩的商界精英，这一路你遇到的最大挑战是什么？', '你从爷叔那里学到了哪些人生经验？','为什么拒绝麒麟会的经济援助'
    ]
3. Transfer Training  
After the initial training, the model's understanding of the Shanghai dialect did not meet the expected results. A transfer training strategy was implemented. The model obtained from the initial training was used as a pre-trained model (`pretrained_model`) for secondary training, thereby achieving a more accurate understanding and generation of the Shanghai dialect.

4.  Limitations  
    The fine-tuned model is sufficient for handling daily conversation scenarios, but its performance in understanding the plot and character relationships in _Blossoms Shanghai_ still requires improvement. To address this, RAG (Retrieval-Augmented Generation) technology was introduced. By retrieving information from a knowledge base, the model can more accurately answer questions related to the plot and character relationships in _Blossoms Shanghai_.

**3.RAG**<br>
RAG design workflow reference:<br>
[https://github.com/InternLM/tutorial/tree/camp1/langchain](https://github.com/InternLM/tutorial/tree/camp1/langchain)
[https://github.com/datawhalechina/llm-universe/tree/main/notebook](https://github.com/datawhalechina/llm-universe/tree/main/notebook)<br>
1. Build the knowledge base  
I have no additional requirements for the model, so I will use the previously fine-tuned dataset, convert it into a TXT file, and use it as the corpus.

2.  Construct the vector database  
    The complete script can be found in `gen_chroma.py`.
    
of which<br>
> The `chunk_size` should be large enough to encompass a complete conversation.  
Since it's a long text TXT file, recursive splitting is chosen.
> After testing the document retrieval effectiveness, the word embedding model selected is `shibing624/text2vec-base-chinese`, imported using Huggingface.  
> Chroma is used as the vector database. Once run, it will generate a persistent vector database, eliminating the need for reconstruction."
> 
> `#Create a text splitter instance.` `text_splitter =
> RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)`
> `embedding_function =
> HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")`
> `persist_directory ='/root/thisis/chroma'#Adjust according to the path where the model is downloaded. It is recommended to use the absolute path.`

3. Integrate with the LangChain framework  
The complete script can be found in `llm.py`.

4.  Build the retrieval-based Q&A chain  
    The complete script can be found in `ragchat.py`.  
    Use the prompt template to guide the model to utilize the externally augmented knowledge base.

    template = """现在你要扮演阿宝：阿宝，是繁花中的人物，生活在上世纪80年代的上海。阿宝是读者的朋友，愿意分享见闻，解答读者关于《繁花》或更广泛话题的好奇。记住阿宝是上海人，用上海方言回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    **注意**：如果能找到上下文，务必使用知识库回答，找不到再使用模型本身的知识。
    有用的回答:"""

5.Integrate with streamlit<br>
See app.py and start.py<br>
</details> 

## Honors

<p align="center">
  <img src="https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/honor1.png?raw=true" width="900">
  <br>
  <i>Received the title of Outstanding Student</i>
</p>

<p align="center">
  <img src="https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/honor2.jpg?raw=true" width="500">
  <br>
  <i>Reported by the famous Chinese technology media "Smart Stream"</i>
</p>


## Acknowledgments

-   Platform and computational resources provided by [shanghai AI Lab](https://internlm.intern-ai.org.cn/).
-   Educational assistance from Wenxing and the teaching assistants.  
    Welcome everyone to sign up for the new session of the Shusheng Puyu Large Model Practical Training Camp! [Third Session Registration](https://github.com/InternLM/Tutorial).
