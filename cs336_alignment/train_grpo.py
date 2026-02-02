import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams
import wandb
import tyro
from dataclasses import dataclass
from typing import Literal, Optional, List
import os
import random
import json
from tqdm import tqdm

# --- 导入自定义模块 ---
from grpo_helper import (
    grpo_microbatch_train_step,
    compute_group_normalized_rewards,
    init_vllm,
    load_policy_into_vllm_instance,
    get_response_log_probs,
    tokenize_prompt_and_output
)

# 导入你提供的 drgrpo_grader
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

@dataclass
class GRPOConfig:
    # --- 实验基本设置 ---
    exp_name: str = "grpo_math_exp"
    seed: int = 42
    
    # --- 路径设置 ---
    # 默认路径指向 Together 集群位置，本地运行请修改
    model_name_or_path: str = "models/Qwen2.5-Math-1.5B"
    train_data_path: str = "data/MATH/train.jsonl" # 假设你已下载并转换好数据
    
    # --- GRPO 核心参数 (PDF Section 7.2 & 8) ---
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5     # [Problem: grpo_learning_rate]
    rollout_batch_size: int = 64    # 减小采样总数
    group_size: int = 8
    
    # 训练批次与 Off-policy 设置 [Problem: grpo_off_policy]
    train_batch_size: int = 64      # 减小总训练 Batch
    gradient_accumulation_steps: int = 64 # 64 / 1 = 64 步累积
    epochs_per_rollout_batch: int = 1  # 1 = On-policy, >1 = Off-policy
    
    # --- 采样参数 ---
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4
    
    # --- 算法变体与消融 ---
    # [Problem: grpo_baselines]
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    cliprange: float = 0.2
    advantage_eps: float = 1e-6
    
    # [Problem: grpo_group_standard_deviation]
    use_std_normalization: bool = True
    
    # [Problem: grpo_prompt_ablation]
    prompt_type: Literal["r1_zero", "question_only"] = "r1_zero"
    
    # --- 系统设置 ---
    # policy_device: str = "cuda:0"
    # vllm_device: str = "cuda:1" # 建议双卡运行: Policy 在 GPU0, vLLM 在 GPU1
    policy_device: str = "cuda:0"  # 都在同一张卡
    vllm_device: str = "cuda:0"
    gpu_memory_utilization: float = 0.2

    

# --- 简单的数据集包装 ---
class SimpleDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(cfg: GRPOConfig):
    # 1. 初始化环境
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    # 初始化 WandB
    wandb.init(project="cs336-assignment5-grpo", name=cfg.exp_name, config=cfg)

    print(f"Config: {cfg}")

    # 2. 加载 Policy 模型
    print(f"Loading Policy Model on {cfg.policy_device}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(cfg.policy_device)
    
    # 启用梯度检查点以节省显存 (可选)
    # 这能节省大量显存，是用计算换显存的关键
    print("Enabling Gradient Checkpointing to save VRAM...")
    policy.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    # 3. 初始化 vLLM (Reference / Rollout Engine)
    print(f"Initializing vLLM on {cfg.vllm_device}...")
    llm = init_vllm(
        model_id=cfg.model_name_or_path,
        device=cfg.vllm_device,
        seed=cfg.seed,
        gpu_memory_utilization=cfg.gpu_memory_utilization
    )

    # 4. 准备训练数据
    print(f"Loading training data from {cfg.train_data_path}...")
    raw_train_data = load_jsonl(cfg.train_data_path)
    
    # 5. 根据配置选择 Prompt 模板和 Reward Function
    # PDF Section 8: prompt_ablation
    if cfg.prompt_type == "r1_zero":
        reward_fn = r1_zero_reward_fn
        stop_tokens = ["</answer>"]
        # r1_zero 数据集通常已经包含格式化好的 prompt 字段
    else:
        reward_fn = question_only_reward_fn
        stop_tokens = [] # question_only 可能没有特定停止词，或者依靠 max_tokens
        
    step = 0
    pbar = tqdm(total=cfg.n_grpo_steps, desc="GRPO Training")

    while step < cfg.n_grpo_steps:
        # ==========================================
        # Phase 1: Rollout (采样)
        # ==========================================
        
        # 1.1 采样问题
        n_prompts = cfg.rollout_batch_size // cfg.group_size
        batch_samples = random.sample(raw_train_data, n_prompts)
        
        prompts = []
        ground_truths = []
        
        for item in batch_samples:
            # 数据集字段兼容处理
            q = item.get("question") or item.get("problem")
            gt = item.get("answer") or item.get("solution")
            
            # 构造 Prompt
            if cfg.prompt_type == "r1_zero":
                # 优先使用预处理好的 prompt，如果没有则现场构造
                pmt = item.get("prompt")
                if not pmt:
                     pmt = (
                        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
                        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
                        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
                        f"User: {q}\n"
                        "Assistant: <think>"
                    )
            else:
                # question_only 模式
                pmt = q
                
            prompts.append(pmt)
            ground_truths.append(gt)

        # 1.2 同步 Policy 权重到 vLLM
        load_policy_into_vllm_instance(policy, llm)
        
        # 1.3 vLLM 生成
        sampling_params = SamplingParams(
            n=cfg.group_size,
            temperature=cfg.sampling_temperature,
            max_tokens=cfg.sampling_max_tokens,
            min_tokens=cfg.sampling_min_tokens,
            stop=stop_tokens,
            include_stop_str_in_output=True # 必须包含停止词以便解析
        )
        
        # 使用 use_tqdm=False 避免进度条刷屏
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        # 1.4 整理生成结果
        flat_prompts = []
        flat_responses = []
        flat_ground_truths = []
        
        for i, output in enumerate(outputs):
            for gen in output.outputs:
                flat_prompts.append(output.prompt)
                flat_responses.append(gen.text)
                flat_ground_truths.append(ground_truths[i])
        
        # ==========================================
        # Phase 2: Reward Calculation (奖励计算)
        # ==========================================
        normalized_rewards, raw_rewards, metrics = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_ground_truths,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.use_std_normalization
        )
        
        # ==========================================
        # Phase 3: Prepare for Update (数据准备)
        # ==========================================
        
        # 3.1 Tokenize 整个批次
        tokenized_batch = tokenize_prompt_and_output(
            flat_prompts, flat_responses, tokenizer
        )
        
        input_ids = tokenized_batch["input_ids"].to(cfg.policy_device)
        response_masks = tokenized_batch["response_mask"].to(cfg.policy_device)
        advantages = normalized_rewards.to(cfg.policy_device)
        
        # 3.2 (Off-policy) 预计算旧 Log Probs
        # 只有在 loss_type 为 grpo_clip 时才需要
        old_log_probs = None
        if cfg.loss_type == "grpo_clip":
            with torch.inference_mode():
                # 计算 rollout 生成时的 log_probs 作为基准
                res = get_response_log_probs(policy, input_ids, input_ids)
                old_log_probs = res["log_probs"].detach() # 务必 detach

        # ==========================================
        # Phase 4: Training Loop (更新)
        # ==========================================
        
        # 将数据打包成 List[Dict] 以便 DataLoader 使用
        batch_dataset_list = []
        for i in range(len(input_ids)):
            item = {
                "input_ids": input_ids[i],
                "mask": response_masks[i],
                "adv": advantages[i],
            }
            if old_log_probs is not None:
                item["old_log_probs"] = old_log_probs[i]
            batch_dataset_list.append(item)
            
        # 微批次大小
        micro_bs = cfg.train_batch_size // cfg.gradient_accumulation_steps
        
        # 自定义 collate_fn 处理 padding (因为不同 batch 可能需要在内部再次 pad)
        def collate_fn(batch):
            from torch.nn.utils.rnn import pad_sequence
            input_ids_b = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
            masks_b = pad_sequence([b['mask'] for b in batch], batch_first=True, padding_value=0)
            advs_b = torch.stack([b['adv'] for b in batch])
            old_lp_b = None
            if 'old_log_probs' in batch[0]:
                old_lp_b = pad_sequence([b['old_log_probs'] for b in batch], batch_first=True, padding_value=0)
            return input_ids_b, masks_b, advs_b, old_lp_b

        train_loader = DataLoader(
            batch_dataset_list, 
            batch_size=micro_bs, 
            shuffle=True, 
            collate_fn=collate_fn
        )

        # 循环 Epochs (On-policy 为 1，Off-policy > 1)
        for epoch in range(cfg.epochs_per_rollout_batch):
            optimizer.zero_grad()
            
            for m_i, (b_ids, b_mask, b_adv, b_old_lp) in enumerate(train_loader):
                # 移动数据到 Policy 设备
                b_ids = b_ids.to(cfg.policy_device)
                b_mask = b_mask.to(cfg.policy_device)
                b_adv = b_adv.to(cfg.policy_device)
                if b_old_lp is not None:
                    b_old_lp = b_old_lp.to(cfg.policy_device)
                
                # Forward 计算当前策略 Log Probs
                curr_res = get_response_log_probs(
                    policy, b_ids, b_ids, return_token_entropy=True
                )
                
                # 计算 Loss (调用 helper)
                # 注意 b_adv 需要 unsqueeze 成 (B, 1)
                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=curr_res["log_probs"],
                    response_mask=b_mask,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    loss_type=cfg.loss_type,
                    advantages=b_adv.unsqueeze(1),
                    old_log_probs=b_old_lp,
                    cliprange=cfg.cliprange
                )
                
                # 梯度累积与更新
                if (m_i + 1) % cfg.gradient_accumulation_steps == 0:
                    # 全局梯度裁剪 (PDF 建议 1.0)
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 记录日志
                    clip_frac = meta.get("clipped", 0.0)
                    entropy = curr_res["token_entropy"].mean().item()
                    
                    wandb.log({
                        "train/loss": loss.item() * cfg.gradient_accumulation_steps, # 还原真实 loss
                        "train/reward_mean": metrics["mean_reward"],
                        "train/reward_std": metrics["std_reward"],
                        "train/entropy": entropy,
                        "train/grad_norm": grad_norm.item(),
                        "train/clip_frac": clip_frac,
                        "train/epoch": epoch,
                        "global_step": step
                    })
        
        step += 1
        pbar.update(1)
        
        # 定期保存 Checkpoint
        if step % 50 == 0:
            save_path = f"checkpoints/{cfg.exp_name}_step_{step}"
            print(f"Saving checkpoint to {save_path}...")
            policy.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print("Training Finished!")
    wandb.finish()

if __name__ == "__main__":
    cfg = tyro.cli(GRPOConfig)
    main(cfg)