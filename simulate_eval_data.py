import numpy as np
import pandas as pd
import anndata
import os

def generate_circle_coords(n, radius=10, center=(0, 0)):
    """生成均匀分布在圆形内部的空间坐标"""
    r = radius * np.sqrt(np.random.uniform(0, 1, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return pd.DataFrame({'x': x, 'y': y})

def _nb_rvs(mu, theta, rng):
    """负二项采样：Poisson-Gamma 混合"""
    mu = np.asarray(mu, dtype=float)
    theta_arr = np.broadcast_to(theta, mu.shape).astype(float)
    lam = rng.gamma(shape=np.maximum(theta_arr, 1e-8),
                    scale=np.maximum(mu, 1e-12) / np.maximum(theta_arr, 1e-8))
    return rng.poisson(lam)

def simulate_benchmark_data(n=3000, n_levels=10, output_file="benchmark_data.h5ad"):
    """
    生成用于评估SVG方法的基准数据。
    包含4组基因，每组 n_levels 个（默认10个）。
    
    变量控制矩阵：
    - 纵向（1-10）：信号强度(amp) 线性递减，模拟从清晰到模糊。
    - 横向（Group A vs B vs C）：同一编号的基因，amp, base, theta 完全一致。
    """
    
    # 1. 基础设置
    rng = np.random.default_rng(42) # 固定种子以复现
    coord_df = generate_circle_coords(n)
    coord_df['barcode'] = [f"spot_{i}" for i in range(n)]
    coord_df.set_index('barcode', inplace=True)
    coords = coord_df[['x', 'y']].values
    
    # 估计半径用于归一化尺度
    radius_est = np.sqrt((coords**2).sum(axis=1)).max()
    
    # 2. 定义梯度参数 (纵向变化)
    # 信号强度从 10.0 (强) 降到 0.5 (极弱，接近背景波动)
    amp_levels = np.linspace(10.0, 0.5, n_levels)
    
    # 固定参数 (保持不变)
    base_mean = 0.5   # 背景表达量
    theta_val = 10.0  # 离散度 (Overdispersion)
    
    # 形状宽度参数 (sigma)，控制图案的"粗细"
    # 统一设为半径的 15%，保证不同形状的特征尺度一致
    sigma = 0.15 * radius_est 
    
    # 3. 初始化数据容器
    # 4组: Stripe, Radial, Ring, Null
    total_genes = n_levels * 4
    counts = np.zeros((n, total_genes), dtype=int)
    gene_names = []
    
    # -------------------------------------------------
    # Group 1: Stripe (条带型) - 垂直穿过圆心 x=0
    # -------------------------------------------------
    for i, amp in enumerate(amp_levels):
        # 计算距离轴线的距离
        dist = np.abs(coords[:, 0]) # 距离 y 轴的距离
        # 高斯轮廓 [0, 1]
        pattern = np.exp(-dist**2 / (2 * sigma**2))
        
        # 添加乘性噪音 (模拟捕获效率差异)
        efficiency = rng.gamma(shape=20.0, scale=1/20.0, size=n)
        mu = (base_mean + amp * pattern) * efficiency
        
        col_idx = i
        counts[:, col_idx] = _nb_rvs(mu, theta_val, rng)
        gene_names.append(f"Stripe_{i+1}_amp{amp:.1f}")

    # -------------------------------------------------
    # Group 2: Radial (点状扩散) - 位于圆心 (0,0)
    # -------------------------------------------------
    for i, amp in enumerate(amp_levels):
        # 计算距离圆心的距离
        dist = np.sqrt(np.sum(coords**2, axis=1))
        # 高斯轮廓 [0, 1]
        pattern = np.exp(-dist**2 / (2 * sigma**2))
        
        efficiency = rng.gamma(shape=20.0, scale=1/20.0, size=n)
        mu = (base_mean + amp * pattern) * efficiency
        
        col_idx = n_levels + i
        counts[:, col_idx] = _nb_rvs(mu, theta_val, rng)
        gene_names.append(f"Radial_{i+1}_amp{amp:.1f}")

    # -------------------------------------------------
    # Group 3: Ring (环状) - 位于半径 0.5 处
    # -------------------------------------------------
    target_radius = 0.5 * radius_est
    for i, amp in enumerate(amp_levels):
        # 计算距离目标环半径的距离
        dist_from_center = np.sqrt(np.sum(coords**2, axis=1))
        dist = np.abs(dist_from_center - target_radius)
        # 高斯轮廓 [0, 1]
        pattern = np.exp(-dist**2 / (2 * sigma**2))
        
        efficiency = rng.gamma(shape=20.0, scale=1/20.0, size=n)
        mu = (base_mean + amp * pattern) * efficiency
        
        col_idx = 2 * n_levels + i
        counts[:, col_idx] = _nb_rvs(mu, theta_val, rng)
        gene_names.append(f"Ring_{i+1}_amp{amp:.1f}")

    # -------------------------------------------------
    # Group 4: Null (阴性对照) - 无空间模式
    # -------------------------------------------------
    for i, amp in enumerate(amp_levels):
        # pattern 恒为 0
        pattern = np.zeros(n)
        
        # 注意：Null组虽然没有pattern，但为了公平对比FDR，
        # 我们保持 base_mean 和 theta 一致。
        # amp 在这里不起作用，因为 pattern 是 0。
        
        efficiency = rng.gamma(shape=20.0, scale=1/20.0, size=n)
        mu = base_mean * efficiency 
        
        col_idx = 3 * n_levels + i
        counts[:, col_idx] = _nb_rvs(mu, theta_val, rng)
        gene_names.append(f"Null_{i+1}")

    # 4. 保存数据
    adata = anndata.AnnData(
        X=counts,
        obs=coord_df,
        var=pd.DataFrame(index=gene_names)
    )
    adata.obsm['spatial'] = coords
    
    # 添加元数据方便后续分析
    adata.var['pattern_type'] = ['Stripe']*10 + ['Radial']*10 + ['Ring']*10 + ['Null']*10
    adata.var['signal_amp'] = list(amp_levels) * 3 + [0.0]*10 # Null amp is effectively 0
    adata.var['rank_id'] = list(range(1, 11)) * 4

    save_path = os.path.join(os.path.dirname(__file__), output_file)
    adata.write_h5ad(save_path)
    print(f"基准测试数据已生成: {save_path}")
    print(f"包含基因: {len(gene_names)} 个 (4组 x 10个等级)")
    print(f"强度梯度: {amp_levels}")

if __name__ == "__main__":
    simulate_benchmark_data()