# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F

from dataset import TrajectoryDataset
from predictor_model import PredictorTransformer

# --- 超参数和配置 ---
ZARR_PATH = 'opt13b_trajectories_2t'
BATCH_SIZE = 8  # 根据你的GPU显存调整
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_SPLIT = 0.1 # 10%的数据用于验证

# OPT-13B Specifics (从数据生成脚本中获取)
CHUNK_SIZE = 2000
NUM_LAYERS = 40
D_MODEL = 5120
NUM_HEADS = 40
HEAD_DIM = 128

def attention_kldiv_loss(predicted_query_proxy, true_queries, true_keys, temperature=1.0):
    """
    计算预测的注意力分布与真实的注意力分布之间的KL散度损失。

    Args:
        predicted_query_proxy (Tensor): 模型的输出, shape [B, Num_Layers, Head_Dim]
        true_queries (Tensor): 真实的查询向量, shape [B, Num_Layers, Seq_Len, Num_Heads, Head_Dim]
        true_keys (Tensor): 真实的键向量, shape [B, Num_Layers, Seq_Len, Num_Heads, Head_Dim]
        temperature (float): Softmax的温度系数，用于平滑分布。

    Returns:
        Tensor: 计算出的KL散度损失 (标量)
    """
    # 我们关心的是最后一个token的查询向量与所有历史key的交互
    # true_queries_last_token: [B, Num_Layers, Num_Heads, Head_Dim]
    true_queries_last_token = true_queries[:, :, -1, :, :]

    # 为了简化，我们将所有头的意图进行平均，作为该层的"真实"查询意图。
    # 这是一个合理的设计选择，可以后续迭代（例如，尝试预测每个头）。
    # true_query_avg_heads: [B, Num_Layers, Head_Dim]
    true_query_avg_heads = true_queries_last_token.mean(dim=2) # 在heads维度上平均
    
    # 计算真实的注意力分数 (logits)
    # 使用einsum进行高效的批处理矩阵乘法
    # b: batch, l: layer, s: sequence_len, h: head, d: head_dim
    # true_scores: [B, Num_Layers, Seq_Len]
    true_scores = torch.einsum('bld,blshd->bls', true_query_avg_heads, true_keys)

    # 计算预测的注意力分数 (logits)
    # predicted_query_proxy与所有key交互
    # predicted_scores: [B, Num_Layers, Seq_Len]
    predicted_scores = torch.einsum('bld,blsd->bls', predicted_query_proxy, keys_avg_heads)

    # 使用KL散度损失
    # KLDiv(log_softmax(pred), softmax(true))
    # log_softmax(pred) 是更数值稳定的计算方式
    log_pred_dist = F.log_softmax(predicted_scores / temperature, dim=-1)
    true_dist = F.softmax(true_scores / temperature, dim=-1)
    
    # `kl_div`期望的输入是 (input, target)，这里是 (log_pred, true_dist)
    # reduction='batchmean' 会对batch和所有其他维度求平均
    loss = F.kl_div(log_pred_dist, true_dist, reduction='batchmean', log_target=False)
    
    return loss

def main():
    print(f"Using device: {DEVICE}")

    # 1. 数据加载
    full_dataset = TrajectoryDataset(ZARR_PATH)
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Data loaded: {train_size} training samples, {val_size} validation samples.")

    # 2. 模型初始化
    model = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_LAYERS,
        opt_head_dim=HEAD_DIM
    ).to(DEVICE)
    
    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for embeddings, queries, keys in pbar:
            embeddings = embeddings.to(DEVICE)
            queries = queries.to(DEVICE)
            keys = keys.to(DEVICE)

            optimizer.zero_grad()
            
            # 前向传播
            predicted_proxy = model(embeddings)
            
            # 计算损失
            loss = attention_kldiv_loss(predicted_proxy, queries, keys)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
            for embeddings, queries, keys in pbar_val:
                embeddings = embeddings.to(DEVICE)
                queries = queries.to(DEVICE)
                keys = keys.to(DEVICE)
                
                predicted_proxy = model(embeddings)
                loss = attention_kldiv_loss(predicted_proxy, queries, keys)
                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        
        # 保存模型
        torch.save(model.state_dict(), f'predictor_epoch_{epoch+1}.pth')

    print("--- Training Finished ---")

if __name__ == '__main__':
    main()