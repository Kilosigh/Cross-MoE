import torch
import h5py
import numpy as np
import os

class HDF5DataController:
    def __init__(self, db_path="time_mmd_features.h5", d_model=768):
        self.db_path = db_path
        self.d_model = d_model
        self._init_db()

    def _init_db(self):
        """如果数据库不存在，则初始化一个空的 HDF5 文件"""
        if not os.path.exists(self.db_path):
            with h5py.File(self.db_path, "w") as f:
                # 创建可无限扩展的数据集 (maxshape=None)
                f.create_dataset("text_embeddings", 
                                 shape=(0, self.d_model), 
                                 maxshape=(None, self.d_model), 
                                 dtype='float16')
                f.create_dataset("timestamps", 
                                 shape=(0,), 
                                 maxshape=(None,), 
                                 dtype='S20') # 存储时间戳字符串
            print(f"✅ 初始化空 HDF5 特征库: {self.db_path}")

    def append_new_text_feature(self, timestamp: str, text_vector: torch.Tensor):
        """
        模拟实时推理时：将新收到并由 LLM Encode 好的单条向量落盘缓存
        text_vector: shape (1, d_model)
        """
        vector_np = text_vector.detach().cpu().numpy().astype(np.float16)
        
        with h5py.File(self.db_path, "a") as f:
            emb_ds = f["text_embeddings"]
            ts_ds = f["timestamps"]
            
            curr_size = emb_ds.shape[0]
            # 扩容
            emb_ds.resize((curr_size + 1, self.d_model))
            ts_ds.resize((curr_size + 1,))
            
            # 写入新数据
            emb_ds[curr_size:] = vector_np
            ts_ds[curr_size] = timestamp.encode('utf8')

    def retrieve_historical_window(self, lookback_steps: int) -> torch.Tensor:
        """
        模拟系统回测/对齐时：利用内存映射(Zero-copy)高速拉取历史区间特征
        """
        with h5py.File(self.db_path, "r") as f:
            emb_ds = f["text_embeddings"]
            total_len = emb_ds.shape[0]
            
            if total_len == 0:
                return torch.zeros((0, self.d_model), dtype=torch.float16)
                
            start_idx = max(0, total_len - lookback_steps)
            # 直接从磁盘切片读取到内存，HDF5 底层是 C 连续的二进制拷贝，极快
            numpy_data = emb_ds[start_idx:total_len]
            
        # 转换为 Tensor，准备送入 GPU
        return torch.from_numpy(numpy_data).to(torch.float16)

# ==========================================
# 测试演示
# ==========================================
if __name__ == "__main__":
    db = HDF5DataController()
    
    # 1. 模拟第一天的新闻 Encode 并落盘
    print("模拟写入特征...")
    dummy_day1 = torch.randn((1, 768), dtype=torch.float16)
    db.append_new_text_feature("2026-04-01", dummy_day1)
    
    dummy_day2 = torch.randn((1, 768), dtype=torch.float16)
    db.append_new_text_feature("2026-04-02", dummy_day2)
    
    # 2. 模拟系统查询近 7 天的数据，送入 Triton 算子前
    print("模拟高频拉取...")
    hist_tensor = db.retrieve_historical_window(lookback_steps=7)
    print(f"成功拉取历史张量，形状: {hist_tensor.shape}, 设备: {hist_tensor.device}")