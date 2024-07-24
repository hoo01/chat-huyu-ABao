# 🌷🤵chat-沪语-阿宝
[![license][license-image]][license-url] | [![OpenXLab_Model][OpenXLab_Model-image]][OpenXLab_Model-url] | [<img src="./images/modelscope_logo.png" width="20px" /> ModelScope][ModelScope-url]

[license-image]: ./images/license.svg
[OpenXLab_Model-image]: https://cdn-static.openxlab.org.cn/header/openxlab_models.svg
[license-url]: ./LICENSE
[OpenXLab_Model-url]: https://openxlab.org.cn/models/detail/hoo01/chat-huyu-ABao
[ModelScope-url]: https://www.modelscope.cn/models/hooo01/chat-huyu-ABao

## 简介

chat-沪语-阿宝是基于《繁花》剧本和大模型生成的问答对，使用InternLM2进行Lora微调+RAG语料库得到的扮演阿宝的上海方言角色聊天模型。

> 阿宝，一个土生土长的上海人，在九十年代上海商界中摸爬滚打的人物。
> 
> 生于普通家境，祖辈资产阶级背景。1966年家庭遭变，阿宝不甘平凡，从里弄工厂工人起步，渴望改写命运。
> 
> 爷叔，商界老将，成为阿宝导师兼挚友，传授市场智慧，助力其商途。初恋雪芝，公交偶遇，深情难敌现实，十年重逢方知受骗。汪小姐，外贸精英，既是事业推手，亦有情感牵绊，阿宝却始终避谈感情，留下遗憾。玲子，日本归沪女子，助阿宝解困，共营“夜东京”。李李，黄河路头牌饭店“至真园”的老板，经历曲折，终入佛门。
> 
> 宝总在与麒麟会在股市较量中陷入困境，其服饰公司股价大跌，面临平仓风险。蔡总未履行股票承诺，宝总资金被套。宝总拒绝撤资或求助麒麟会，相信麒麟会为了自救会出手相助。麒麟会提出帮助，但宝总拒绝了现金和法人股的回报，只要求麒麟会撤销对川沙农地的收购，保护当地农民利益，并决定退出股市。宝总从和平饭店的七年经历中领悟到人生起伏，能接受从繁华到平凡的转变，坚信土地与未来同在。

## 架构图
![enter image description here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/%E6%9E%B6%E6%9E%84.jpg?raw=true)
 ## 项目亮点

 1. 使用上海方言回答
 2.  《繁花》关键剧情、人物关系等问题上回答表现良好

## 快速开始

❕ 暂时仅提供本地部署方法，建议使用50%以上的A100、cuda11.7的配置运行

**1.clone 本项目至本地开发机** 

    git clone [https://github.com/hoo01/chat-huyu-ABao.git](https://github.com/hoo01/chat-huyu-ABao.git)

**2.配置环境**

    #创建虚拟环境
    conda create -n fanhua python=3.10 
    # 激活环境 
    conda activate fanhua
    # 安装所需依赖（这一步所需时间较长）
    cd chat-huyu-ABao  
    pip install -r requirements.txt

**3.生成RAG依赖的Chroma数据库**

    python gen_chroma.py

**4.启动(此部分因下载模型需要较长时间)**

    python start.py

> 默认端口和ip为127.0.0.1:7860，如需变更请打开start.py修改<br>
> 非本地开发机，需要做端口映射<br>
> 以浦语InternStudio开发机为例，打开本地计算机的命令提示符，输入
> `ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p <自己的号码>`<br>
> password：输入自己的密码 ![enter image description
> here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/%E6%98%A0%E5%B0%841.png?raw=true)
> ![enter image description
> here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/%E6%98%A0%E5%B0%842.png?raw=true)

**5.示例效果**
![enter image description here](https://github.com/hoo01/chat-huyu-ABao/blob/main/imgs/test1.jpg?raw=true)

## 微调思路
<details>
<summary>点击展开详细思路</summary>

**1.数据准备**<br>
数据分为两个部分：《繁花》剧本和大模型生成的问答对<br>
1.《繁花》台词转换成xtuner的训练格式 <br>
数据集格式、环境配置见xtuner官方教学文档：<br>
[https://github.com/InternLM/Tutorial/blob/camp2/data_fine_tuning/data_fine_tuning.md](https://github.com/InternLM/Tutorial/blob/camp2/data_fine_tuning/data_fine_tuning.md)
转换后的台词格式如下：

    `{"conversation": [{"system": "阿宝", "input": "", "output": "如果做生意是一门艺术的话，这个人绝对是个老法师，他叫爷叔是我人生中的第一个贵人，我认识他的时候，他刚从提篮桥监狱出来。"}, {"system": "阿宝", "input": "爷叔：谁啊。", "output": "爷叔好，我是阿宝。你不认识我了？"}, {"system": "阿宝", "input": "今天的太阳晒不到明天的衣裳的，时间，时间决定一切。回去吧。", "output": "爷叔我还没讲完呢。我是来跟你学生意的。"}, {"system": "阿宝", "input": "你知道什么叫生意？", "output": "这个不很简单的，生意嘛就是一买一卖，将本求利。"}, {"system": "阿宝", "input": "这是你爷爷告诉你的？你爷爷的爷爷是地主，到你爷爷这代是资本家，那么你呢。", "output": "我呢就是……"}, {"system": "阿宝", "input": "你是没钱开公司，那你原始积累怎么办，靠偷靠抢。", "output": "所以我……"}, {"system": "阿宝", "input": "所以你的原始积累只能靠借，所以你的账一开始就是负数，如果你要借一百块做生意，利息至少要百分之二十。你要做到多少利润，这生意可以继续下去。", "output": "一百二十块？"}, {"system": "阿宝", "input": "错，两百八十块。", "output": "为什么？"}, {"system": "阿宝", "input": "本金加利息一百二十块，生活二十块，成本四十块，剩下的一百块可以做生意了。一百块的成本，两百八十块的利润，什么概念？暴利，你知道我什么地方出来的。", "output": "你这个不是投机倒把？"}, {"system": "阿宝", "input": "当然不是，你对现在的形势了解吗？对现在的政策研究过吗？什么钱好赚，什么钱不可以赚了，赚了要吃官司的。拍拍胸脯，就要发财了，想也不要想。回去，回去。", "output": "爷叔，我是要做外贸的。"}, {"system": "阿宝", "input": "你懂外语？", "output": "外语可以学，我阿哥，在香港开公司，我可以做他的营业代表，现在流行“三来一补”，我觉得可以做的。"}, {"system": "阿宝", "input": "外贸就是借人家的鸡生你自己的蛋，不过人家凭什么要把鸡借给你，帮你生蛋呢？好，这样，明天你到这个地方租一间房间，到明天中午没有你的消息，我们两个就算拉倒。", "output": "和平饭店。"}]}`

2.使用API生成问答对<br>
2.1使用大模型API，提供prompt，批量生成问题<br>
完整脚本见data/数据准备/gen_q_api.ipynb<br>
2.2使用大模型API，提供prompt，让大模型扮演阿宝批量生成回答<br>
完整脚本见data/数据准备/q2a_api.ipynb<br>
3.使用API将以上两部分的output转化为上海方言<br>
完整脚本见data/数据准备/pth2huyu.py<br>

通过以上3步得到符合xtuner微调格式的jsonl数据。<br>

**2.微调模型**<br>
xtuner微调工具包的官方教程：  <br>
https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md
https://github.com/InternLM/Tutorial/blob/camp2/data_fine_tuning/data_fine_tuning.md<br>
1.选择基座模型<br>
通过多次测试，同样的参数配置下，7b模型对上海方言的学习效果显著优于1.8b，因此基座模型选择了internlm2-chat-7b。<br>
2.配置文件修改<br>
按照教程里的配置文件，对PART1修改，其余部分未动：<br>
part1改动：

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
3.迁移训练<br>
在初步训练完成后，模型对于上海方言的掌握尚未达到预期效果。实施了迁移训练策略。将初次训练得到的模型作为预训练模型（pretrained_model），进行二次训练，从而实现对上海方言更为精准的理解和生成。<br>
4.局限<br>
微调后的模型足以应对日常对话场景，但对《繁花》的剧情和人物关系理解方面，其表现仍有待提升。对此，引入RAG（检索增强生成）技术。通过检索知识库信息，辅助模型更准确地回答《繁花》情节和人物关系问题。

**3.RAG检索增强**<br>
RAG设计链路参考：<br>
[https://github.com/InternLM/tutorial/tree/camp1/langchain](https://github.com/InternLM/tutorial/tree/camp1/langchain)
[https://github.com/datawhalechina/llm-universe/tree/main/notebook](https://github.com/datawhalechina/llm-universe/tree/main/notebook)<br>
1.知识库搭建<br>
我对模型没有额外需求，还是用之前微调的数据集，转成txt文件作为语料库。<br>
2.构建向量数据库<br>
完整脚本见gen_chroma.py<br>
其中<br>
> chunk_size的大小要能包含一个完整的conversation； 因为是长文本txt，分割选择递归分割；<br>
> 经过测试召回文档的效果，词向量模型最终选择的是shibing624/text2vec-base-chinese，使用huggingface导入；<br>
> 使用chroma作为向量数据库，运行即可得到持久化的向量数据库，无需重复构建。
> 
> `#创建文本分割器实例` `text_splitter =
> RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)`
> `embedding_function =
> HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")`
> `persist_directory ='/root/thisis/chroma'#根据下载模型的路径调整，建议写绝对路径`

3.接入LangChain框架<br>
完整脚本见llm.py<br>
4.构建检索问答链<br>
完整脚本见ragchat.py<br>
在prompt template引导模型使用外部增强的知识库

    template = """现在你要扮演阿宝：阿宝，是繁花中的人物，生活在上世纪80年代的上海。阿宝是读者的朋友，愿意分享见闻，解答读者关于《繁花》或更广泛话题的好奇。记住阿宝是上海人，用上海方言回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    **注意**：如果能找到上下文，务必使用知识库回答，找不到再使用模型本身的知识。
    有用的回答:"""

5.接入streamlit<br>
见app.py和start.py<br>
</details> 

## 致谢

 - [书生浦语](https://internlm.intern-ai.org.cn/)提供的平台及算力资源 
 - 闻星大佬和助教提供的教学帮助
欢迎大家报名新一期的书生浦语大模型实战营！[第三期报名](https://github.com/InternLM/Tutorial)

