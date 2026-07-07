# 核心思路是：三类方案结合 -- 关键词+长度+正则（零成本）、FAISS 向量库（语义泛化能力）、小模型分类（准确率高）
# 用规则拦确定的，用向量处理不确定的
# 实际效果：无路由全走GPT5.5 Token = 100% ； 混合路由 Token = 28%  额外延迟 + 10ms  用户满意度 - 1%  路由给轻量模型比率 = 82%

# 第一阶段：正则快筛（O(1)，<1ms），目的是确定性请求 零成本方式拦截 60%-70% 
import re
from typing import Optional

def fast_route(query: str) -> Optional[str]:
    # 1. 长度阈值：超长 query 直接走强模型
    if len(query) > 800:
        return "heavy"  # 强模型
    
    # 2. 关键词命中：明确的复杂任务直接路由
    heavy_keywords = ["代码", "推理", "分析", "对比", "为什么", "如何优化"]
    if any(k in query for k in heavy_keywords):
        return "heavy"
    
    # 3. 句式识别：问数场景下的确定性路由
    if re.match(r"^(查询|查看|统计).*(本月|上周|今年)", query):
        return "light"  # 轻量模型
    
    return None  # 无法确定，进入第二阶段


# 第二阶段：向量精排（~10ms），思路是用 FAISS 做语义相似度匹配，类似KNN分类算法（在初期没有足够样本时，用规则路由兜底，until 500+ 样本）
import faiss
import numpy as np
from typing import Dict

def semantic_route(query: str, index: faiss.Index, route_examples: dict, threshold: float = 0.5) -> str:
    # 1. 将 query 转为 embedding
    q_embedding = embed_model.encode(query)
    
    # 2. 在 FAISS 索引中检索最相似的 K 个样本
    distances, indices = index.search(q_embedding, k=5)
    
    # 3. 投票决策
    votes = {"light": 0, "heavy": 0}
    for idx, dist in zip(indices[0], distances[0]):
        if dist > threshold:  # 距离太远，不参与投票
            continue
        label = route_examples[idx]["label"]
        votes[label] += 1
    
    # 4. 多数投票 + 置信度兜底
    if max(votes.values()) >= 3:
        return max(votes, key=votes.get)
    return "heavy"  # 无法确定时走保守路线
