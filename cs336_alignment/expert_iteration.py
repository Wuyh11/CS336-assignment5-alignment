import os
import json
import torch
import random
import argparse
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from vllm import LLM, SamplingParams

# 引用现有辅助函数
from cs336_alignment.sft_helper import tokenize_prompt_and_output, sft_microbatch_train_step, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# 1. Dataset Class
# -----------------------------------------------------------------------------
class GSM8KQuestionDataset(Dataset):
    """只用于采样问题的 Dataset"""
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} questions from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SFTDataset(Dataset):
    """用于 SFT 训练的 Dataset"""
    def __init__(self, data_list):
        self.data = data_list # list of {"question": ..., "reasoning": ..., "final_answer": ...}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant\n"
            "solves it. The Assistant first thinks about the reasoning process in the mind and\n"
            "then provides the User with the answer. The reasoning process is enclosed within\n"
            "<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,\n"
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            "\n"
            f"User: {item['question']}\n"
            "Assistant: <think>\n"
        )
        # 注意：严格匹配 grader 格式，空格而不是换行
        output = f"{item['reasoning']}\n</think> <answer>{item['final_answer']}</answer>"
        return prompt, output

# -----------------------------------------------------------------------------
# 2. Worker Functions (Executed in separate processes)
# -----------------------------------------------------------------------------

def _vllm_generate_worker(model_path, questions, output_file, g_rollouts, seed):
    """vLLM 生成子进程"""
    set_seed(seed)
    # 显存优化：预留一些给 PyTorch 上下文，但 vLLM 主要吃显存
    # A10 24G 比较紧凑，设为 0.8 或 0.7 以防万一，但既然是独立进程，0.85 也可以尝试
    gpu_memory_utilization = 0.85 
    
    logger.info(f"Worker: Loading vLLM model from {model_path}...")
    try:
        llm = LLM(
            model=model_path,
            device="cuda",
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
            seed=seed
        )
    except Exception as e:
        logger.error(f"vLLM Init failed: {e}")
        raise e
    
    prompts = []
    meta_data = [] 
    
    prompt_template = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant\n"
        "solves it. The Assistant first thinks about the reasoning process in the mind and\n"
        "then provides the User with the answer. The reasoning process is enclosed within\n"
        "<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,\n"
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        "\n"
        "User: {question}\n"
        "Assistant: <think>\n"
    )

    for q_item in questions:
        p = prompt_template.format(question=q_item['question'])
        prompts.append(p)
        meta_data.append(q_item)

    logger.info(f"Worker: Generating {len(prompts)} * {g_rollouts} responses...")
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        min_tokens=4,
        n=g_rollouts,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        q_item = meta_data[i]
        for sample in output.outputs:
            generated_text = sample.text
            results.append({
                "question": q_item['question'],
                "gold_answer": q_item['answer'],
                "generated_text": generated_text
            })
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    
    logger.info("Worker: Generation finished.")


def _train_worker(queue, model_path, train_data, output_dir, epochs, lr, batch_size, grad_acc):
    """SFT 训练子进程"""
    set_seed(42)
    try:
        logger.info(f"Worker: Training SFT on {len(train_data)} samples...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        train_dataset = SFTDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        
        optimizer = AdamW(model.parameters(), lr=lr)
        model.train()
        
        total_entropy = 0.0
        steps_logged = 0
        
        for epoch in range(epochs):
            for step, batch in enumerate(train_loader): # 不用 tqdm 避免多进程打印混乱
                prompts = [x[0] for x in batch]
                outputs = [x[1] for x in batch]
                
                tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
                input_ids = tokenized["input_ids"].to(model.device)
                labels = tokenized["labels"].to(model.device)
                response_mask = tokenized["response_mask"].to(model.device)
                
                log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                policy_log_probs = log_probs_dict["log_probs"]
                
                if "token_entropy" in log_probs_dict:
                    entropy_tensor = log_probs_dict["token_entropy"]
                    valid_tokens = response_mask.sum()
                    if valid_tokens > 0:
                        batch_mean_entropy = (entropy_tensor * response_mask).sum() / valid_tokens
                        total_entropy += batch_mean_entropy.item()
                        steps_logged += 1

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    gradient_accumulation_steps=grad_acc
                )
                
                if (step + 1) % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        
        logger.info(f"Worker: Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        mean_entropy = total_entropy / max(1, steps_logged)
        queue.put(mean_entropy)
        
    except Exception as e:
        logger.error(f"Train Worker Error: {e}")
        queue.put(None) # Signal failure
        raise e


def _eval_worker(queue, model_path, test_data_path, sample_size):
    """评估子进程"""
    set_seed(42)
    try:
        logger.info(f"Worker: Evaluating model {model_path}...")
        
        # 关键修正：padding_side="left" 用于生成
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        test_questions = []
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_questions.append(json.loads(line))
                    
        if sample_size and sample_size < len(test_questions):
            test_questions = random.sample(test_questions, sample_size)
            
        correct = 0
        total = len(test_questions)
        batch_size = 4
        
        prompt_template = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant\n"
            "solves it. The Assistant first thinks about the reasoning process in the mind and\n"
            "then provides the User with the answer. The reasoning process is enclosed within\n"
            "<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,\n"
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            "\n"
            "User: {question}\n"
            "Assistant: <think>\n"
        )

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch = test_questions[i:i+batch_size]
                prompts = [prompt_template.format(question=q['question']) for q in batch]
                gts = [q['answer'] for q in batch]
                
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                input_len = inputs.input_ids.shape[1]
                gen_texts = tokenizer.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)
                
                for gen_text, gt in zip(gen_texts, gts):
                    full_res = f"<think>\n{gen_text}"
                    if "####" in gt:
                        gt_val = gt.split("####")[-1].strip()
                    else:
                        gt_val = gt.strip()
                        
                    score = r1_zero_reward_fn(full_res, gt_val)
                    if score['answer_reward'] > 0:
                        correct += 1
                        
        acc = correct / total
        logger.info(f"Worker: Evaluation Accuracy: {acc:.4f}")
        queue.put(acc)
        
    except Exception as e:
        logger.error(f"Eval Worker Error: {e}")
        queue.put(None)
        raise e

# -----------------------------------------------------------------------------
# 3. Process Runners
# -----------------------------------------------------------------------------

def run_generation(model_path, batch_questions, g_rollouts, step_idx, output_dir, seed):
    output_file = os.path.join(output_dir, f"gen_step_{step_idx}.json")
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_vllm_generate_worker,
        args=(model_path, batch_questions, output_file, g_rollouts, seed)
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"vLLM generation worker failed with exit code {p.exitcode}")
    
    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(model_path, test_data_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_eval_worker,
        args=(queue, model_path, test_data_path, 200)
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Evaluation worker failed")
    
    result = queue.get()
    if result is None:
        raise RuntimeError("Evaluation worker returned None (error occurred)")
    return result

def run_training(model_path, train_data, output_dir, epochs, lr):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    # SFT Batch Size settings
    bs = 4
    grad_acc = 4
    
    p = ctx.Process(
        target=_train_worker,
        args=(queue, model_path, train_data, output_dir, epochs, lr, bs, grad_acc)
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Training worker failed")
    
    result = queue.get()
    if result is None:
        raise RuntimeError("Training worker returned None (error occurred)")
    return result

# -----------------------------------------------------------------------------
# 4. Filtering
# -----------------------------------------------------------------------------
def filter_correct_samples(raw_generations):
    correct_samples = []
    for item in raw_generations:
        question = item['question']
        gold_ans = item['gold_answer']
        gen_text = item['generated_text']
        
        full_response = f"<think>\n{gen_text}"
        if "####" in gold_ans:
            gold_val = gold_ans.split("####")[-1].strip()
        else:
            gold_val = gold_ans.strip()
            
        score = r1_zero_reward_fn(full_response, gold_val)
        
        if score["answer_reward"] == 1.0:
            if "</think>" in gen_text and "<answer>" in gen_text:
                parts = gen_text.split("</think>")
                reasoning = parts[0].strip()
                ans_part = parts[1]
                if "<answer>" in ans_part and "</answer>" in ans_part:
                    final_answer = ans_part.split("<answer>")[1].split("</answer>")[0].strip()
                    correct_samples.append({
                        "question": question,
                        "reasoning": reasoning,
                        "final_answer": final_answer
                    })
    return correct_samples

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/ei_experiment")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--ei_steps", type=int, default=5)
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.data_dir, "train.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")
    
    all_train_questions = GSM8KQuestionDataset(train_path).data
    current_model_path = args.base_model
    
    logs = {"steps": [], "accuracy": [], "entropy": []}
    
    logger.info("Running initial evaluation (isolated process)...")
    init_acc = run_evaluation(current_model_path, test_path)
    logger.info(f"Initial Accuracy: {init_acc:.4f}")
    
    logs["steps"].append(0)
    logs["accuracy"].append(init_acc)
    logs["entropy"].append(0.0)
    
    for step in range(1, args.ei_steps + 1):
        logger.info(f"=== Expert Iteration Step {step}/{args.ei_steps} ===")
        step_dir = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        
        # 1. Sample
        db_size = min(args.batch_size, len(all_train_questions))
        batch_questions = random.sample(all_train_questions, db_size)
        
        # 2. Generate
        logger.info(f"Generating rollouts...")
        raw_generations = run_generation(
            current_model_path, 
            batch_questions, 
            args.rollouts, 
            step, 
            step_dir,
            seed=42+step
        )
        
        # 3. Filter
        sft_data = filter_correct_samples(raw_generations)
        logger.info(f"Filtered {len(sft_data)} correct samples.")
        
        if len(sft_data) == 0:
            logger.warning("No correct samples! Skipping SFT.")
            mean_entropy = 0.0
        else:
            # 4. Train
            next_model_path = os.path.join(step_dir, "model")
            mean_entropy = run_training(
                current_model_path,
                sft_data,
                next_model_path,
                epochs=args.sft_epochs,
                lr=1e-5
            )
            current_model_path = next_model_path
            
        # 5. Eval
        acc = run_evaluation(current_model_path, test_path)
        
        logs["steps"].append(step)
        logs["accuracy"].append(acc)
        logs["entropy"].append(mean_entropy)
        
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(logs, f, indent=4)
            
    logger.info("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()