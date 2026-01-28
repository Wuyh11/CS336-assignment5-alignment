import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable, Any
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
import wandb # 假设你使用 wandb 进行日志记录

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """
    对 prompt 和 output 进行分词，并构建 input_ids, labels 和 response_mask。
    response_mask 在 response 的 token 位置为 1，在 prompt 或 padding 位置为 0。
    """
    assert len(prompt_strs) == len(output_strs)

    # 分别对 prompt 和 output 进行分词（不添加 special tokens，手动拼接更可控）
    # 注意：根据模型不同，是否添加 BOS 需要自行处理，这里假设传入的字符串已经处理好或者模型不需要
    encoded_prompts = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    encoded_outputs = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    input_ids_list = []
    labels_list = []
    masks_list = []

    for p_ids, o_ids in zip(encoded_prompts, encoded_outputs):
        # 拼接 prompt 和 output
        full_ids = p_ids + o_ids
        
        # 转换为 Tensor
        full_tensor = torch.tensor(full_ids, dtype=torch.long)
        
        # 构建 input_ids 和 labels (shift logic)
        # input_ids: 0 到 L-2
        # labels: 1 到 L-1
        # mask: 对应 labels 的位置。如果 labels[i] 是 response 的一部分，则 mask[i]=1
        
        # 完整的 token 序列长度
        seq_len = len(full_ids)
        
        # 我们的目标是预测 output 部分。Output 从 len(p_ids) 开始。
        # labels 的第 i 个元素对应原序列的第 i+1 个 token。
        # 我们希望 i+1 >= len(p_ids) 时 mask 为 1。
        # 即 i >= len(p_ids) - 1。
        
        mask = torch.zeros(seq_len - 1, dtype=torch.long) # 使用 long 或 float 方便计算
        start_mask_idx = len(p_ids) - 1
        
        # 处理边界情况：如果 prompt 为空，则从 0 开始；如果 prompt 很长，从 prompt 结束前一位开始
        if start_mask_idx < 0:
            mask[:] = 1
        else:
            mask[start_mask_idx:] = 1
            
        input_ids_list.append(full_tensor[:-1])
        labels_list.append(full_tensor[1:])
        masks_list.append(mask)

    # Padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # 使用 pad_sequence 进行 padding，batch_first=True
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=pad_id)
    # mask padding 值为 0
    masks_padded = pad_sequence(masks_list, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "response_mask": masks_padded
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 预测分布的熵。
    H(p) = - sum(p * log(p))
    """
    # logits: (batch_size, sequence_length, vocab_size)
    # 使用 log_softmax 保证数值稳定性
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    # 熵 = - sum(p * log_p)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy # (batch_size, sequence_length)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> Dict[str, torch.Tensor]:
    """
    获取给定输入的 log probabilities (以及可选的 entropy)。
    """
    # 前向传播
    # 注意：这里假设 model 已经处理好 device 放置
    outputs = model(input_ids)
    logits = outputs.logits # (batch_size, seq_len, vocab_size)

    # 计算所有词表的 log_softmax
    log_probs_all = F.log_softmax(logits, dim=-1)

    # 获取目标 label 对应的 log_prob
    # labels: (batch_size, seq_len)
    # gather 需要 index 维度匹配，unsqueeze 扩展最后一维
    # gather result: (batch_size, seq_len, 1) -> squeeze 回 (batch_size, seq_len)
    log_probs = torch.gather(log_probs_all, -1, labels.unsqueeze(-1)).squeeze(-1)

    result = {
        "log_probs": log_probs
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    对 tensor 进行求和并归一化，同时考虑 mask。
    """
    # 确保 mask 和 tensor 形状兼容或者可以广播
    masked_tensor = tensor * mask
    
    # 求和
    if dim is not None:
        summed = torch.sum(masked_tensor, dim=dim)
    else:
        summed = torch.sum(masked_tensor)
        
    # 归一化
    return summed / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    执行 SFT 的单个 microbatch 训练步。
    """
    # 1. 计算加权和 / 常数
    # masked_normalize 已经处理了 sum / normalize_constant
    total_log_prob = masked_normalize(
        policy_log_probs, 
        response_mask, 
        normalize_constant=normalize_constant, 
        dim=None
    )
    
    # 2. 获取 batch size
    batch_size = policy_log_probs.shape[0]

    # 3. 计算最终 Loss
    # 注意：根据测试快照，我们需要同时除以 batch_size 和 gradient_accumulation_steps
    # 这意味着我们计算的是 "Global Mean Loss"
    loss = -total_log_prob / (batch_size * gradient_accumulation_steps)
    
    # 4. 反向传播
    loss.backward()
    
    # 5. 返回 Loss
    # 注意：测试要求返回的也是 scaled loss
    return loss, {"loss": loss.detach()}

def log_generations(
    vllm_model: Any, # 假设是 vLLM engine 或兼容对象
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    step: int,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    sampling_params: Optional[Any] = None
) -> None:
    """
    生成样本并记录到 WandB 或控制台。
    """
    # 1. 生成
    # 注意：这里假设使用 vLLM 的 generate 接口，它通常返回 RequestOutput 列表
    outputs = vllm_model.generate(prompts, sampling_params)
    
    table_data = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_entropy = 0.0
    total_len = 0.0
    correct_len = 0.0
    incorrect_len = 0.0
    num_correct = 0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        # 如果 vLLM 返回了 logprobs，可以用来计算 entropy，否则跳过
        # 这里假设不做复杂的 entropy 计算以简化，或者需要在 sampling_params 中开启 logprobs
        
        # 2. 计算 Reward
        rewards = reward_fn(generated_text, ground_truths[i])
        r, fr, ar = rewards.get("reward", 0), rewards.get("format_reward", 0), rewards.get("answer_reward", 0)
        
        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar
        current_len = len(generated_text) # 或者用 token 长度
        total_len += current_len
        
        if ar > 0: # 假设 answer_reward > 0 为正确
            correct_len += current_len
            num_correct += 1
        else:
            incorrect_len += current_len

        # 添加到表格数据
        table_data.append([
            prompts[i],
            generated_text,
            ground_truths[i],
            r,
            fr,
            ar
        ])

    # 计算统计数据
    n = len(prompts)
    metrics = {
        "eval/mean_reward": total_reward / n,
        "eval/mean_format_reward": total_format_reward / n,
        "eval/mean_answer_reward": total_answer_reward / n,
        "eval/avg_len": total_len / n,
        "eval/avg_len_correct": correct_len / num_correct if num_correct > 0 else 0,
        "eval/avg_len_incorrect": incorrect_len / (n - num_correct) if (n - num_correct) > 0 else 0,
        "eval_step": step
    }

    # 3. 记录
    if wandb.run is not None:
        wandb.log(metrics)
        # 记录一个简单的表格（比如前10个例子）
        columns = ["Prompt", "Generated", "Ground Truth", "Reward", "Format Reward", "Answer Reward"]
        wandb.log({"eval/generations": wandb.Table(columns=columns, data=table_data[:20])})
    
    print(f"Step {step} Eval: Reward={metrics['eval/mean_reward']:.4f}, Format={metrics['eval/mean_format_reward']:.4f}")