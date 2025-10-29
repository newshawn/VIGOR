import os
from src.open_r1.intuitor_trainer import INTUITORTrainer
from src.open_r1.configs import GRPOConfig

# 准备一份最小化 config，保证必要字段齐全
cfg = GRPOConfig()
cfg.semantic_embedding_api_base = "https://api.siliconflow.cn/v1"  # 如果需要 /v1，就改成 .../v1
cfg.semantic_embedding_api_key = os.getenv("SEMANTIC_EMBEDDING_API_KEY") or "sk-pkkoqekaidbszexjuwchtvgybvllcftpdxbfdydwzezyklxi"
cfg.semantic_embedding_model = "Qwen/Qwen3-Embedding-0.6B"
cfg.semantic_embedding_batch_size = 4
cfg.semantic_similarity_low = 0.2
cfg.semantic_similarity_high = 0.9

# 构造一个假的 trainer，只用到 embedding 功能
trainer = object.__new__(INTUITORTrainer)
trainer.semantic_embedding_api_base = cfg.semantic_embedding_api_base
trainer.semantic_embedding_api_key = cfg.semantic_embedding_api_key
trainer.semantic_embedding_model = cfg.semantic_embedding_model
trainer.semantic_embedding_timeout = 30.0
trainer.semantic_embedding_batch_size = cfg.semantic_embedding_batch_size

# 伪造 completion 文本和 mask
texts = [
    "解释牛顿第一定律。",
    "列出三种常见的数据结构并说明用途。",
    "讲一个关于机器学习的笑话。",
    "如果地球自转停止会发生什么？"
]
import torch
mask = torch.ones(len(texts), 10, dtype=torch.long)  # 随便给个非零 mask
import pdb; pdb.set_trace()
embeddings, valid_mask = trainer._compute_completion_embeddings_remote(texts, mask)

print("valid:", valid_mask)
print("embedding shape:", embeddings.shape)
print("前两个向量示例：")
print(embeddings[:2])
