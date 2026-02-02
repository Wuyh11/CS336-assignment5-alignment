from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
from vllm import LLM
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    计算每个响应的组归一化奖励。

    参数:
        reward_fn: 奖励函数，用于评分响应与真值的匹配。
        rollout_responses: 由模型生成的响应列表。
        repeated_ground_truths: 每个问题的真值答案列表。
        group_size: 每个问题的响应数（即每组的大小）。
        advantage_eps: 为避免除零错误而加的小常数。
        normalize_by_std: 如果为True，使用标准差归一化；否则仅使用均值归一化。

    返回:
        tuple: 包含(归一化奖励, 原始奖励, 元数据)的元组
            - 归一化奖励: 每个响应的归一化奖励。
            - 原始奖励: 每个响应的原始奖励。
            - 元数据: 包含奖励的统计信息（如均值、标准差、最小值、最大值）。
    """
    
    # 步骤 1: 计算原始奖励
    raw_rewards = []
    for i in range(0, len(rollout_responses), group_size):
        group_responses = rollout_responses[i:i + group_size]
        group_truths = repeated_ground_truths[i:i + group_size]
        
        # 为当前组的响应计算奖励
        rewards = []
        for response, truth in zip(group_responses, group_truths):
            reward = reward_fn(response, truth)  # 假设reward_fn返回一个包含"reward"的字典
            rewards.append(reward['reward'])  # 从字典中提取奖励值
        raw_rewards.extend(rewards)
    
    raw_rewards = torch.tensor(raw_rewards)
    
    # 步骤 2: 归一化奖励
    num_groups = len(rollout_responses) // group_size
    normalized_rewards = []
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        group_raw_rewards = raw_rewards[start_idx:end_idx]
        
        # 计算均值和标准差
        group_mean = group_raw_rewards.mean()
        group_std = group_raw_rewards.std()
        
        if normalize_by_std:
            # 使用标准差归一化
            normalized_group = (group_raw_rewards - group_mean) / (group_std + advantage_eps)
        else:
            # 只使用均值归一化
            normalized_group = group_raw_rewards - group_mean
        
        normalized_rewards.extend(normalized_group)
    
    normalized_rewards = torch.tensor(normalized_rewards)
    
    # 步骤 3: 收集元数据
    metadata = {
        'mean_reward': raw_rewards.mean().item(),
        'std_reward': raw_rewards.std().item(),
        'min_reward': raw_rewards.min().item(),
        'max_reward': raw_rewards.max().item(),
    }
    
    return normalized_rewards, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: # 修改返回类型
    """
    计算每个token的策略梯度损失。
    """
    # 确保 raw_rewards_or_advantages 是 (batch_size, 1)
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)
        
    # 自动广播 (Broadcasting) 会处理 (batch_size, 1) * (batch_size, seq_len)
    loss = -raw_rewards_or_advantages * policy_log_probs
    
    return loss, {}  # 返回一个空的 dict 作为 metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算每个token的GRPO-Clip损失。

    参数：
        advantages: (batch_size, 1) 的张量，每个响应的优势 A。
        policy_log_probs: (batch_size, sequence_length) 的张量，当前策略下每个token的log-probabilities。
        old_log_probs: (batch_size, sequence_length) 的张量，旧策略下每个token的log-probabilities。
        cliprange: 剪切参数 ϵ，控制更新幅度。

    返回：
        tuple: 包含两个部分：
            1. loss: (batch_size, sequence_length) 的张量，表示每个token的GRPO-Clip损失。
            2. metadata: 包含关于每个token是否被剪切的信息。
    """
    
    # 计算每个token的比例
    ratio = torch.exp(policy_log_probs - old_log_probs)
    
    # 计算每个token的剪切损失
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)  # 计算GRPO-Clip损失
    
    # 计算剪切标志
    clipped = (ratio > (1 + cliprange)) | (ratio < (1 - cliprange))
    
    # 返回损失和剪切信息
    metadata = {
        'clipped': clipped.float().mean().item()  # 计算被剪切的token比例
    }
    
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards必须提供"
        # 这里现在可以安全解包了
        loss, metadata = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages必须提供"
        loss, metadata = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss, metadata


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    # 将mask和tensor相乘，确保无效的位置被置为0
    masked_tensor = tensor * mask
    
    if dim is not None:
        # 在指定维度上进行求和
        sum_masked = masked_tensor.sum(dim=dim)
        # 计算mask中为1的元素个数
        count_masked = mask.sum(dim=dim)
    else:
        # 如果没有指定dim，计算所有元素的和
        sum_masked = masked_tensor.sum()
        count_masked = mask.sum()
    
    # 返回归一化后的均值
    return sum_masked / count_masked

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    在微批次（microbatch）上执行前向和反向传播 [cite: 805]。
    """
    
    # 步骤 1: 调用包装器计算每个 token 的原始损失 [cite: 755]
    # compute_policy_gradient_loss 会处理广播 (broadcasting) 和类型分发 [cite: 743, 754]
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 步骤 2: 使用 masked_mean 对 response 区域的 loss 进行聚合 [cite: 786]
    # 这里 dim=1 表示对序列长度维度取均值，只计算 response 内部 token 的平均损失 [cite: 762, 776]
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    
    # 步骤 3: 计算整个 microbatch 的平均损失 [cite: 786]
    # 并除以梯度累积步数，以便后续 backward() 时累加的梯度是正确的平均梯度 [cite: 226, 346]
    loss = per_example_loss.mean() / gradient_accumulation_steps

    # 步骤 4: 执行反向传播 [cite: 817]
    loss.backward()

    # 返回调整后的 loss 标量用于日志记录 [cite: 814]
    return loss, metadata


def grpo_train_loop(
    policy,
    vllm_instance,
    reward_fn,
    train_questions,
    tokenizer,
    optimizer,
    n_grpo_steps: int = 200,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    epochs_per_rollout_batch: int = 1,
    loss_type: str = "reinforce_with_baseline",
    cliprange: float = 0.2,
    advantage_eps: float = 1e-6,
    use_std_normalization: bool = True,
    max_grad_norm: float = 1.0,
):
    # 计算一些核心步数参数 [cite: 845, 849, 857]
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_batch = rollout_batch_size // group_size

    for step in range(n_grpo_steps):
        # --- 步骤 1: 采样 (Rollout Phase) --- [cite: 614]
        # 1.1 从训练集中抽取一批问题 [cite: 432]
        batch_questions = random.sample(train_questions, n_prompts_per_batch)
        
        # 1.2 同步当前策略权重到 vLLM 实例 
        load_policy_into_vllm_instance(policy, vllm_instance)

        # 1.3 为每个问题生成 G 个响应 [cite: 595, 614]
        # 这里需要使用 r1_zero prompt，并在 </answer> 处停止 
        rollout_data = generate_rollouts(vllm_instance, batch_questions, group_size)

        # --- 步骤 2: 奖励与优势计算 (Advantage Phase) --- [cite: 614]
        # 2.1 计算组归一化奖励 [cite: 598, 599, 644]
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_data['responses'],
            repeated_ground_truths=rollout_data['ground_truths'],
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )

        # 2.2 如果是 off-policy (GRPO-Clip)，预先计算旧策略的 log-probs [cite: 866, 867, 982]
        with torch.inference_mode():
            old_log_probs = get_response_log_probs(
                policy, rollout_data['input_ids'], rollout_data['labels']
            )['log_probs']

        # --- 步骤 3: 策略更新 (Update Phase) --- [cite: 614, 976]
        # 准备 DataLoader 进行多轮 epoch 更新
        dataset = RolloutDataset(rollout_data, advantages, old_log_probs)
        dataloader = DataLoader(dataset, batch_size=micro_train_batch_size, shuffle=True)

        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            
            for m_idx, micro_batch in enumerate(dataloader):
                # 执行微批次训练步 [cite: 787, 805]
                # 此函数内部会调用 loss.backward() [cite: 817]
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=get_current_log_probs(policy, micro_batch),
                    response_mask=micro_batch['response_mask'],
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=micro_batch['advantages'],
                    old_log_probs=micro_batch['old_log_probs'],
                    cliprange=cliprange
                )

                # 当累积到足够步数时进行参数更新 [cite: 234, 235]
                if (m_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm) # [cite: 406, 862]
                    optimizer.step()
                    optimizer.zero_grad()

        # --- 步骤 4: 日志记录与验证 --- [cite: 572, 863, 871]
        # 定期进行验证集评估并记录训练奖励
        log_metrics(step, reward_metadata, metadata)

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算每个token的熵 [cite: 272]"""
    # logits: (B, L, V)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor, # 这里实际上主要用到 input_ids 进行 forward
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    """获取 Log Probabilities 和 Entropy [cite: 290]"""
    # model forward
    outputs = model(input_ids)
    logits = outputs.logits # (B, L, V)

    # Shift logits and input_ids to align prediction
    # logits[..., :-1, :] 预测 input_ids[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Gather log probs of the actual tokens
    # log_softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # gather: 选取真实 token 对应的 log_prob
    # shift_labels.unsqueeze(-1) shape: (B, L-1, 1)
    target_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # 为了保持形状一致性 (B, L)，通常会在最后补 0 或在最前补 0，
    # 这里我们为了对齐输入长度，在最后补一个 0 (最后一个token没有下一个token可预测)
    # 或者根据 grpo_helper 的逻辑，通常 mask 会处理对齐。
    # 这里简单补齐到 (B, L)
    zeros = torch.zeros(target_log_probs.shape[0], 1, device=target_log_probs.device)
    target_log_probs = torch.cat([target_log_probs, zeros], dim=1)

    result = {"log_probs": target_log_probs}

    if return_token_entropy:
        entropy = compute_entropy(logits) # (B, L)
        result["token_entropy"] = entropy

    return result

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """初始化 vLLM，包含 PDF 中的 Monkeypatch [cite: 367]"""
    vllm_set_random_seed(seed)
    
    # Monkeypatch to support single-device vLLM alongside PyTorch policy
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=2048,  # <---【新增】强制限制最大长度为 2048 (甚至 1024)
        )
    return llm

def load_policy_into_vllm_instance(policy, llm: LLM):
    """将 Policy 权重同步到 vLLM [cite: 392]"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer, max_length=2048):
    """将 prompt 和 output 拼接并 tokenize，生成 mask [cite: 250]"""
    # 简单实现，实际需处理 padding
    input_ids = []
    response_masks = []
    
    for p, o in zip(prompt_strs, output_strs):
        # 分别编码
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        o_ids = tokenizer.encode(o, add_special_tokens=False)
        
        # 拼接
        full_ids = p_ids + o_ids + [tokenizer.eos_token_id]
        
        # Mask: Prompt 部分为 0，Response 部分为 1
        mask = [0] * len(p_ids) + [1] * len(o_ids) + [1] # EOS 算作 response
        
        # 截断或填充逻辑 (略，假设 batch 内长度不一，使用 pad_sequence)
        input_ids.append(torch.tensor(full_ids, dtype=torch.long))
        response_masks.append(torch.tensor(mask, dtype=torch.long))
    
    from torch.nn.utils.rnn import pad_sequence
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    masks_padded = pad_sequence(response_masks, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids_padded,
        "labels": input_ids_padded, # SFT/RL 中 labels 通常等于 input_ids
        "response_mask": masks_padded
    }