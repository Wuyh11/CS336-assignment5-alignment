import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from cs336_alignment.sft_helper import tokenize_prompt_and_output, sft_microbatch_train_step, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GSM8KDataset(Dataset):
    def __init__(self, data_path, size=None, filter_correct=False):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            if not line.strip(): continue
            item = json.loads(line)
            if "####" in item["answer"]:
                parts = item["answer"].split("####")
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
                item["reasoning"] = reasoning
                item["final_answer"] = final_answer
                self.data.append(item)
        
        if filter_correct:
            self.data = [d for d in self.data if d.get("final_answer")]

        random.shuffle(self.data)
        
        if size is not None and size < len(self.data):
            self.data = self.data[:size]
            
        print(f"Loaded {len(self.data)} examples from {data_path} (Target Size: {size})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        reasoning = item["reasoning"]
        final_answer = item["final_answer"]
        
        # 构建 R1-Zero 格式的 Prompt
        prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant\n"
            "solves it. The Assistant first thinks about the reasoning process in the mind and\n"
            "then provides the User with the answer. The reasoning process is enclosed within\n"
            "<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,\n"
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            "\n"
            f"User: {question}\n"
            "Assistant: <think>\n"
        )
        
        # === FIX: 严格匹配 grader 的格式 (空格而不是换行) ===
        # 错误写法: f"{reasoning}\n</think>\n<answer>{final_answer}</answer>"
        # 正确写法: 注意 </think> 和 <answer> 之间是空格
        output = f"{reasoning}\n</think> <answer>{final_answer}</answer>"
        
        return prompt, output, item["answer"]

def evaluate(model, tokenizer, test_data, batch_size=4, max_new_tokens=512):
    """
    在 Test 集上评估准确率
    test_data: list of (prompt, output, ground_truth) tuples
    """
    model.eval()
    prompts = [item[0] for item in test_data]
    ground_truths = [item[2] for item in test_data] 
    
    # === FIX: 切换 padding_side 为 left 以支持 generation ===
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    correct_count = 0
    total = len(prompts)
    
    print(f"Evaluating on {total} examples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
            batch_prompts = prompts[i:i+batch_size]
            batch_gts = ground_truths[i:i+batch_size]
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            input_len = inputs.input_ids.shape[1]
            generated_texts = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
            
            for gen_text, gt in zip(generated_texts, batch_gts):
                full_response = f"<think>\n{gen_text}" 
                
                if "####" in gt:
                    gt_val = gt.split("####")[-1].strip()
                else:
                    gt_val = gt.strip()
                
                score = r1_zero_reward_fn(full_response, gt_val)
                if score["answer_reward"] > 0:
                    correct_count += 1
    
    # === FIX: 恢复 padding_side ===
    tokenizer.padding_side = original_padding_side

    accuracy = correct_count / total
    model.train() 
    return accuracy

def train_sft(args, train_dataset, test_dataset, model_path, run_name):
    # Load Tokenizer (SFT 训练默认使用 right padding)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    model.train()
    grad_acc_steps = args.grad_acc_steps
    total_steps = len(train_loader) * args.epochs
    
    progress_bar = tqdm(range(total_steps), desc=f"Training {run_name}")
    optimizer.zero_grad()
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            prompts = [x[0] for x in batch]
            outputs = [x[1] for x in batch]
            
            tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
            input_ids = tokenized["input_ids"].to(model.device)
            labels = tokenized["labels"].to(model.device)
            response_mask = tokenized["response_mask"].to(model.device)
            
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            
            loss, _ = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps=grad_acc_steps
            )
            
            is_update_step = ((step + 1) % grad_acc_steps == 0) or ((step + 1) == len(train_loader))
            if is_update_step:
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item() * grad_acc_steps}) 

    print(f"Running final evaluation for {run_name}...")
    accuracy = evaluate(model, tokenizer, test_dataset)
    print(f"Run {run_name} Final Accuracy: {accuracy:.4f}")
    
    del model
    del optimizer
    torch.cuda.empty_cache()
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to local Qwen2.5-Math-1.5B")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data/gsm8k")
    parser.add_argument("--output_dir", type=str, default="results/sft_experiment")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--grad_acc_steps", type=int, default=8) 
    parser.add_argument("--epochs", type=int, default=1) 
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")

    # Fix: 使用 __getitem__ 获取评估数据
    test_ds_obj = GSM8KDataset(test_path, size=None)
    full_indices = list(range(len(test_ds_obj)))
    sample_indices = random.sample(full_indices, min(200, len(full_indices)))
    eval_dataset = [test_ds_obj[i] for i in sample_indices]

    dataset_sizes = [128, 256, 512, 1024, None] 
    results = {}

    print("=== Starting Task 1: Varying Dataset Sizes ===")
    for size in dataset_sizes:
        size_name = str(size) if size else "Full"
        print(f"\nTraining with Dataset Size: {size_name}")
        train_dataset = GSM8KDataset(train_path, size=size)
        acc = train_sft(args, train_dataset, eval_dataset, args.model_path, run_name=f"size_{size_name}")
        results[size_name] = acc
    
    with open(os.path.join(args.output_dir, "task1_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("Task 1 Results:", results)

    print("\n=== Starting Task 2: Filtered Dataset ===")
    filtered_train_dataset = GSM8KDataset(train_path, size=None, filter_correct=True)
    filtered_size = len(filtered_train_dataset)
    acc_filtered = train_sft(args, filtered_train_dataset, eval_dataset, args.model_path, run_name="filtered_full")
    
    results_task2 = {"filtered_size": filtered_size, "accuracy": acc_filtered}
    with open(os.path.join(args.output_dir, "task2_results.json"), "w") as f:
        json.dump(results_task2, f, indent=4)
    print("Task 2 Results:", results_task2)

if __name__ == "__main__":
    main()