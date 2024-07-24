import os
import torch
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

class InternLM(LLM):   
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        try:
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                               trust_remote_code=True, 
                                                               torch_dtype=torch.float16).cuda()
            self.model.eval()  # 将模型设置为评估模式
            print("完成本地模型的加载")        
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            raise

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        
        # 重写调用函数
        system_prompt = """侬叫阿宝，是《繁花》当中个人物，生活辣上世纪 80 年代个上海。侬是大家个朋友，愿意分享见闻，解答伊拉关于《繁花》或者更加广泛话题个好奇。侬是上海人，用上海方言回答问题。
        """

        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"