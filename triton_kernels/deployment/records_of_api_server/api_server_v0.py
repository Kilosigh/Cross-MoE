import os
import h5py
import numpy as np
import torch
import time
import asyncio
import json
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
import contextlib

# 导入您的底层算子与模型
from my_triton_kernel import TorchCrossMoEMultiHeadAttention, FinalCrossMoEMultiHeadAttention

# ==========================================
# 新增：系统真实运行状态监控探针
# ==========================================
class SystemMonitor:
    def __init__(self):
        self.request_records = [] # 记录单次请求: (timestamp, latency_ms)
        self.metrics_history = [] # 记录每秒的汇总数据
        self.is_running = False

    def add_record(self, latency_ms):
        self.request_records.append((time.time(), latency_ms))

    async def _monitor_loop(self):
        """后台轮询任务：每秒计算一次真实 QPS、延时和 GPU 占用"""
        while self.is_running:
            await asyncio.sleep(1.0)
            now = time.time()
            
            # 取出过去 1 秒内的所有请求记录
            # 清理过期的记录以防内存泄漏
            valid_records = [r for r in self.request_records if now - r[0] <= 1.0]
            self.request_records = [r for r in self.request_records if now - r[0] <= 10.0] 
            
            qps = len(valid_records)
            if qps > 0:
                latencies = [r[1] for r in valid_records]
                avg_lat = float(np.mean(latencies))
                p95_lat = float(np.percentile(latencies, 95)) # 提取长尾 P95
            else:
                avg_lat, p95_lat = 0.0, 0.0

            # 采集真实的 GPU 显存利用率
            mem_allocated = torch.cuda.memory_allocated(DEVICE)
            total_mem = torch.cuda.get_device_properties(DEVICE).total_memory
            gpu_mem_pct = (mem_allocated / total_mem) * 100.0

            metric = {
                "timestamp": now,
                "qps": qps,
                "avg_latency": avg_lat,
                "p95_latency": p95_lat,
                "gpu_mem_pct": gpu_mem_pct
            }
            self.metrics_history.append(metric)
            # print(f"[Monitor] QPS: {qps} | P95 Latency: {p95_lat:.2f}ms | GPU Mem: {gpu_mem_pct:.2f}%")

    def dump_metrics(self, filepath="real_metrics.json"):
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=4)
        print(f"✅ 真实监控数据已落盘至: {filepath}")

monitor = SystemMonitor()


# ==========================================
# 1. 异步 HDF5 数据控制器 (保持不变)
# ==========================================
class AsyncHDF5DataController:
    def __init__(self, db_path="time_mmd_features.h5", d_model=768):
        self.db_path = db_path
        self.d_model = d_model
        self._init_db()

    def _init_db(self):
        if not os.path.exists(self.db_path):
            with h5py.File(self.db_path, "w") as f:
                f.create_dataset("text_embeddings", shape=(0, self.d_model), maxshape=(None, self.d_model), dtype='float16')
            print(f"✅ 初始化空 HDF5 特征库: {self.db_path}")

    def _sync_append_mock_data(self, num_records=2000):
        with h5py.File(self.db_path, "a") as f:
            emb_ds = f["text_embeddings"]
            if emb_ds.shape[0] < num_records:
                print(f"⏳ 正在向磁盘预写入 {num_records} 条历史文本向量...")
                dummy_data = np.random.randn(num_records, self.d_model).astype(np.float16)
                emb_ds.resize((num_records, self.d_model))
                emb_ds[:] = dummy_data

    def _sync_retrieve(self, lookback_steps: int) -> torch.Tensor:
        with h5py.File(self.db_path, "r") as f:
            emb_ds = f["text_embeddings"]
            total_len = emb_ds.shape[0]
            if total_len == 0:
                return torch.zeros((0, self.d_model), dtype=torch.float16)
            
            start_idx = max(0, total_len - lookback_steps)
            numpy_data = emb_ds[start_idx:total_len]
            
        return torch.from_numpy(numpy_data).to(torch.float16).pin_memory()

    async def retrieve_historical_window(self, lookback_steps: int) -> torch.Tensor:
        return await run_in_threadpool(self._sync_retrieve, lookback_steps)


# ==========================================
# 2. 全局配置与应用生命周期
# ==========================================
B = 4
N_q = 5       
N_kv = 1024 * 2     
H = 6
D_IN = 768
D_OUT = 768
M = 4
live_len = 512
DEVICE = 'cuda'

db_controller = AsyncHDF5DataController(d_model=D_IN)

native_model = None
triton_model = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global native_model, triton_model
    
    db_controller._sync_append_mock_data(num_records=2000)

    await db_controller.retrieve_historical_window(2047)
    
    print("🚀 正在系统启动阶段预热模型...")
    native_model = TorchCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(DEVICE).half()
    native_model.eval()

    triton_model = FinalCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(DEVICE).half()
    triton_model.eval()

    with torch.no_grad():
        triton_model.router_q.copy_(native_model.router_q)
        triton_model.router_k.copy_(native_model.router_k)
        triton_model.experts_q.copy_(native_model.experts_q)
        triton_model.experts_k.copy_(native_model.experts_k)
        triton_model.w_v.weight.copy_(native_model.w_v.weight)

    with torch.inference_mode():
        dummy_q = torch.randn((B, N_q, D_IN), dtype=torch.float16, device=DEVICE)
        dummy_kv = torch.randn((B, N_kv, D_IN), dtype=torch.float16, device=DEVICE)
        for _ in range(3):
            _ = native_model(dummy_q, dummy_kv)
            _ = triton_model(dummy_q, dummy_kv)
    torch.cuda.synchronize()
    print("✅ 模型预热与 JIT 编译完成，服务已就绪！")
    
    # 启动监控后台任务
    monitor.is_running = True
    monitor_task = asyncio.create_task(monitor._monitor_loop())
    
    yield
    
    # 关闭服务时落盘监控数据
    monitor.is_running = False
    monitor.dump_metrics("real_metrics.json")
    print("🛑 服务关闭，释放资源...")

app = FastAPI(lifespan=lifespan)

# ==========================================
# 3. 异步并发 API 路由 (加入计时与打点)
# ==========================================
async def prepare_multimodal_tensors():
    
    X_q = torch.randn((B, N_q, D_IN), dtype=torch.float16, device=DEVICE)
    hist_text_cpu = await db_controller.retrieve_historical_window(lookback_steps=N_kv - live_len)
    hist_text_gpu = hist_text_cpu.to(DEVICE, non_blocking=True)
    live_text_cpu = torch.randn((live_len, D_IN), dtype=torch.float16, device=DEVICE)
    full_text_cpu = torch.cat([hist_text_gpu, live_text_cpu], dim=0)
    
    # 注意：你原来这里是拼了一个 512 的live_text，为了对其 N_kv，我改为了1。你可以根据实际逻辑调整。
    X_kv = full_text_cpu.unsqueeze(0).expand(B, -1, -1).to(DEVICE, non_blocking=True)
    return X_q, X_kv

@app.post("/predict/native")
async def predict_native():
    start_time = time.time()
    
    X_q, X_kv = await prepare_multimodal_tensors()
    torch.cuda.synchronize()
    with torch.inference_mode():
        out = native_model(X_q, X_kv)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    monitor.add_record(latency) # 埋点采集
    return {"status": "success", "engine": "native"}

@app.post("/predict/triton")
async def predict_triton():
    start_time = time.time()
    
    X_q, X_kv = await prepare_multimodal_tensors()
    torch.cuda.synchronize()
    with torch.inference_mode():
        out = triton_model(X_q, X_kv)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) * 1000
    monitor.add_record(latency) # 埋点采集
    return {"status": "success", "engine": "triton"}