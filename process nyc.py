# src/data/build_nyc_context.py
import pandas as pd
import numpy as np
import pickle
import os

# --- 配置 ---
# 请确保这些路径与你 config.py 里的设置对应
INPUT_PARQUET = 'processed_nyc.parquet'  # 原始数据
OUTPUT_PARQUET = 'processed_nyc_with_grid.parquet'  # 对应 config.py: NYC_DATA_FILENAME
OUTPUT_CONTEXT = 'context_maps_nyc.pkl'  # 对应 config.py: NYC_CONTEXT_FILENAME

# NYC 边界 (曼哈顿核心区)
BOUNDS = {'min_lon': -74.05, 'max_lon': -73.75, 'min_lat': 40.60, 'max_lat': 40.90}
GRID_SIZE = 0.005


def build_context():
    print(f"1. Loading NYC Parquet data from {INPUT_PARQUET}...")
    if not os.path.exists(INPUT_PARQUET):
        print(f"Error: {INPUT_PARQUET} not found. Please verify the file path.")
        return

    df = pd.read_parquet(INPUT_PARQUET)

    print("2. Computing Grid IDs...")
    lat_indices = ((df['latitude'] - BOUNDS['min_lat']) / GRID_SIZE).astype(int)
    lon_indices = ((df['longitude'] - BOUNDS['min_lon']) / GRID_SIZE).astype(int)
    width = int((BOUNDS['max_lon'] - BOUNDS['min_lon']) / GRID_SIZE) + 1

    df['grid_cell_id'] = lat_indices * width + lon_indices

    print("3. Generating Flattened Context Maps...")
    # 【核心修复】生成扁平的字典结构 {(grid_id, hour, type): value}
    flat_context_map = {}

    # 3.1 Importance (重要性)
    # 统计每个网格的总订单数，并将其复制到所有 24 小时，以便按小时查询时能查到
    importance_series = df.groupby('grid_cell_id').size()
    print("   - Processing Importance...")
    for grid_id, count in importance_series.items():
        for h in range(24):
            # 键格式：(int, int, str)
            flat_context_map[(int(grid_id), int(h), 'importance')] = float(count)

    # 3.2 Density (密度)
    # 统计每个网格每小时的唯一工作者数
    print("   - Processing Density...")
    df['hour'] = df['timestamp'].dt.hour
    density_df = df.groupby(['grid_cell_id', 'hour'])['worker_id'].nunique().reset_index()

    for _, row in density_df.iterrows():
        g_id = int(row['grid_cell_id'])
        h = int(row['hour'])
        val = float(row['worker_id'])  # 这里其实是 nunique 的数量
        flat_context_map[(g_id, h, 'density')] = val

    print(f"4. Saving Context Map to {OUTPUT_CONTEXT}...")
    with open(OUTPUT_CONTEXT, 'wb') as f:
        pickle.dump(flat_context_map, f)

    print(f"5. Saving Updated Parquet to {OUTPUT_PARQUET}...")
    # 移除临时列
    if 'hour' in df.columns:
        df.drop(columns=['hour'], inplace=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("\n✅ Success! Data structure fixed.")
    print(f"Total keys in context map: {len(flat_context_map)}")


if __name__ == '__main__':
    build_context()