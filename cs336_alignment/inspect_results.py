import json
import argparse

def inspect_results(file_path):
    print(f"正在分析文件: {file_path} ...\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 1. 统计分类
    # Category 1: Format=1, Answer=1 (正确)
    correct = [d for d in data if d['scores']['format_reward'] == 1 and d['scores']['answer_reward'] == 1]
    # Category 2: Format=1, Answer=0 (格式对但答案错)
    fmt1_ans0 = [d for d in data if d['scores']['format_reward'] == 1 and d['scores']['answer_reward'] == 0]
    # Category 3: Format=0, Answer=0 (格式错误)
    fmt0 = [d for d in data if d['scores']['format_reward'] == 0]

    print("=== 统计结果 (Part b) ===")
    print(f"总样本数: {len(data)}")
    print(f"(1) 格式正确且答案正确 (Correct): {len(correct)}")
    print(f"(2) 格式正确但答案错误 (Wrong Answer): {len(fmt1_ans0)}")
    print(f"(3) 格式错误 (Format Error): {len(fmt0)}")
    print("=========================\n")

    # 2. 检查格式错误 (Format Reward = 0)
    print("=== 案例分析：格式错误 (Format Reward = 0) ===")
    print("请观察以下案例，判断是模型未遵循指令（如缺少 <answer> 标签），还是解析器过于严格？\n")
    for i, item in enumerate(fmt0[:10]):  # 展示前10个
        print(f"[Case {i+1}]")
        print(f"Ground Truth: {item['ground_truth']}")
        # 打印模型输出的最后 300 个字符，通常答案在这里
        print(f"Model Output (Last 300 chars): ...{item['completion'][-300:]!r}") 
        print("-" * 50)

    # 3. 检查答案错误 (Format Reward = 1, Answer Reward = 0)
    print("\n=== 案例分析：答案错误 (Format=1, Answer=0) ===")
    print("请观察以下案例，判断是模型算错了，还是答案格式不匹配（例如 1/2 vs 0.5）？\n")
    for i, item in enumerate(fmt1_ans0[:10]): # 展示前10个
        print(f"[Case {i+1}]")
        print(f"Ground Truth: {item['ground_truth']}")
        print(f"Model Output (Last 300 chars): ...{item['completion'][-300:]!r}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="runs/math_baseline_gsm8k/results.jsonl")
    args = parser.parse_args()
    inspect_results(args.file)