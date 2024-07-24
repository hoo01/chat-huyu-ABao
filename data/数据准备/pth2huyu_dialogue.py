from openai import OpenAI
from tqdm import tqdm
import json
import os
import dashscope
dashscope.api_key=""#这里使用的是阿里云，换成自己的api_key


# 创建SDK客户端实例
client = OpenAI(
    api_key=dashscope.api_key,   # 写自己的API-KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
)

# 输入和输出文件路径
input_file_path = r'\baozong.jsonl'
output_file_path = r'\baozong_huyu.jsonl'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 读取输入JSONL文件，并使用SDK发送每一行到API
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in tqdm(input_file, desc='转换进度'):
        json_obj = json.loads(line)
        conversation = json_obj["conversation"]
        
        # 遍历对话的每一轮
        for conv in conversation:
            if "output" in conv:  # 确保存在output字段
                original_output = conv["output"]
                
                # 构造包含翻译指令的messages列表
                messages = [
                    {"role": "system", "content": "你是一个上海方言专家，你的任务是将普通话文本翻译成上海方言。请直接给出翻译，不要添加任何前缀或后缀,不要添加任何说明："},
                    {"role": "user", "content": original_output}
                ]
                
                try:
                    # 使用同步方式调用API
                    response = client.chat.completions.create(
                        model='qwen-plus',
                        messages=messages,
                        temperature=0
                    )
                    translated_output = response.choices[0].message.content
                    
                except Exception as e:
                    print(f"翻译失败: {e}")
                    translated_output = "翻译失败"
                
                # 更新当前轮次的对话输出
                conv["output"] = translated_output
        
        # 将翻译后的对象写入输出文件
        output_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print("翻译完成，结果已保存到", output_file_path)
