import asyncio
import aiohttp
import time
import numpy as np

async def fetch(session, url):
    start_time = time.time()
    try:
        async with session.post(url) as response:
            await response.json()
            end_time = time.time()
            return end_time - start_time
    except Exception as e:
        # 捕捉因为 OOM 导致的请求失败
        return -1 

async def run_benchmark(concurrency, url):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for _ in range(concurrency)]
        
        start_time = time.time()
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # 过滤掉失败的请求（比如 OOM）
        success_latencies = [l for l in latencies if l != -1]
        failed_count = concurrency - len(success_latencies)
        
        if not success_latencies:
            return 0.0, float('inf'), failed_count
            
        qps = len(success_latencies) / total_time
        avg_latency = np.mean(success_latencies) * 1000  # 转为 ms
        
        return qps, avg_latency, failed_count

def main():
    # 逐渐升高的并发压力池
    concurrency_levels = [1, 5, 10, 50, 100, 500]
    base_url = "http://127.0.0.1:8000"
    
    print("=== 开始端到端系统并发压测 ===")
    for engine in ["triton", "triton_dynamic", "native"]:
        print(f"\n🚀 测试路由引擎: {engine.upper()}")
        print(f"{'并发数':<10} | {'QPS (吞吐量)':<15} | {'平均时延 (ms)':<15} | {'失败/OOM次数'}")
        print("-" * 65)
        
        url = f"{base_url}/predict/{engine}"
        
        for c in concurrency_levels:
            # 正式测试
            qps, avg_lat, fails = asyncio.run(run_benchmark(c, url))
            print(f"{c:<10} | {qps:<15.2f} | {avg_lat:<15.2f} | {fails}")
            
            # 如果连续大量失败，防止系统直接崩溃，暂停一下
            time.sleep(1)

if __name__ == "__main__":
    main()