import json
import os
from datasets import load_dataset
from tqdm import tqdm

def prepare_math_dataset(repo_id="qwedsacf/competition_math", output_dir="./MATH"):
    print(f"正在从 Hugging Face 下载数据集: {repo_id}...")
    dataset_dict = load_dataset(repo_id)
    
    # 获取可用的分片名称
    available_splits = list(dataset_dict.keys())
    print(f"发现可用分片: {available_splits}")

    r1_zero_template = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        "User: {question}\n"
        "Assistant: <think>"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 逻辑处理：针对 qwedsacf/competition_math 只有一个 train 分片的情况
    if len(available_splits) == 1 and "train" in available_splits:
        full_data = list(dataset_dict["train"])
        if len(full_data) == 12500:
            print("检测到合并数据集，正在按 7500(train)/5000(test) 比例拆分...")
            splits_to_process = {
                'train.jsonl': full_data[:7500],
                'validation.jsonl': full_data[7500:]
            }
        else:
            print(f"数据集大小为 {len(full_data)}，将全部保存为 train.jsonl")
            splits_to_process = {'train.jsonl': full_data}
    else:
        # 如果有多个分片，则按原逻辑匹配
        splits_to_process = {}
        if "train" in dataset_dict:
            splits_to_process['train.jsonl'] = dataset_dict["train"]
        if "test" in dataset_dict:
            splits_to_process['validation.jsonl'] = dataset_dict["test"]
        elif "validation" in dataset_dict:
            splits_to_process['validation.jsonl'] = dataset_dict["validation"]

    # 执行保存
    for filename, data_list in splits_to_process.items():
        output_path = os.path.join(output_dir, filename)
        print(f"正在处理并保存至 {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(data_list):
                # 兼容不同的列名
                question = item.get('problem') or item.get('question')
                answer = item.get('solution') or item.get('answer')
                
                formatted_data = {
                    "prompt": r1_zero_template.format(question=question),
                    "answer": answer
                }
                f.write(json.dumps(formatted_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    prepare_math_dataset()