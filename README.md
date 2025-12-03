# PPF Surface Match Python 使用指南

本文档介绍如何使用PPF (Point Pair Feature) 表面匹配库的Python接口进行3D物体识别和姿态估计。

## 快速开始

### 1. 安装和构建

#### 方法一：使用安装脚本（推荐）

```bash
cd python
./install.sh
```

#### 方法二：手动构建

```bash
cd python
pip install -r requirements.txt
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
cd ..
cp build/ppf.so .
```

### 2. 基本使用

```python
from ppf_wrapper import PPFMatcher, PointCloud

# 加载模型和场景
model = PointCloud.from_file("model.ply")
scene = PointCloud.from_file("scene.ply")

# 创建匹配器并训练模型
matcher = PPFMatcher()
matcher.train_model(model, sampling_distance_rel=0.025)

# 执行匹配
matches = matcher.match_scene(
    scene,
    sampling_distance_rel=0.025,
    key_point_fraction=0.1,
    min_score=0.1,
    num_matches=5
)

# 处理结果
for i, (pose, score) in enumerate(matches):
    print(f"匹配 {i+1}: 分数 = {score:.4f}")
    print("姿态矩阵:")
    print(pose)
```

## 详细API说明

### PointCloud 类

用于表示3D点云数据。

#### 创建点云

```python
# 从PLY文件加载
pc = PointCloud.from_file("model.ply")

# 从NumPy数组创建
import numpy as np
points = np.random.rand(1000, 3).astype(np.float32)  # Nx3点坐标
normals = np.random.rand(1000, 3).astype(np.float32)  # Nx3法向量（可选）
pc = PointCloud.from_numpy(points, normals)
```

#### 属性和方法

```python
# 基本属性
print(f"点数: {pc.num_points}")
print(f"是否有法向量: {pc.has_normals}")

# 设置视点（用于法向量计算）
pc.set_view_point(x, y, z)

# 保存到文件
pc.save("output.ply")

# 转换为NumPy数组
points, normals = pc.to_numpy()
```

### PPFMatcher 类

主要的匹配器类，用于训练模型和执行匹配。

#### 训练模型

```python
matcher = PPFMatcher()

# 基本训练
matcher.train_model(model, sampling_distance_rel=0.025)

# 自定义训练参数
matcher.train_model(
    model,
    sampling_distance_rel=0.025,           # 采样距离（相对于物体直径）
    feat_distance_step_rel=0.04,           # 特征距离步长
    feat_angle_resolution=30,              # 特征角度分辨率
    pose_ref_rel_sampling_distance=0.01,  # 姿态精化采样距离
    knn_normal=10,                         # 法向量估计的最近邻数
    smooth_normal=True                     # 是否平滑法向量
)
```

#### 执行匹配

```python
# 基本匹配
matches = matcher.match_scene(scene)

# 自定义匹配参数
matches = matcher.match_scene(
    scene,
    sampling_distance_rel=0.025,           # 场景采样距离
    key_point_fraction=0.1,                # 关键点比例
    min_score=0.1,                         # 最小分数
    num_matches=5,                         # 最大匹配数
    knn_normal=10,                         # 法向量估计的最近邻数
    smooth_normal=True,                    # 是否平滑法向量
    invert_normal=False,                   # 是否反转法向量
    max_overlap_dist_rel=0.5,              # 最大重叠距离
    sparse_pose_refinement=True,           # 稀疏姿态精化
    dense_pose_refinement=True,            # 密集姿态精化
    pose_ref_num_steps=5,                  # 姿态精化步数
    pose_ref_dist_threshold_rel=0.1,       # 姿态精化距离阈值
    pose_ref_scoring_dist_rel=0.01         # 评分距离阈值
)
```

#### 模型保存和加载

```python
# 保存训练好的模型
matcher.save_model("trained_model.ppf")

# 加载预训练模型
new_matcher = PPFMatcher()
new_matcher.load_model("trained_model.ppf")
```

### 工具函数

```python
from ppf_wrapper import transform_pointcloud, sample_mesh, compute_bounding_box

# 变换点云
transformed = transform_pointcloud(pc, pose_matrix, use_normal=True)

# 采样网格
sampled = sample_mesh(pc, radius=0.01)

# 计算边界框
min_point, max_point = compute_bounding_box(pc)
```

## 完整示例

### 示例1：基本物体识别

```python
#!/usr/bin/env python3

import numpy as np
from ppf_wrapper import PPFMatcher, PointCloud, transform_pointcloud

def main():
    # 加载数据
    model = PointCloud.from_file("gear.ply")
    scene = PointCloud.from_file("gear_n35.ply")
    
    # 设置视点
    model.set_view_point(620, 100, 500)
    scene.set_view_point(-200, -50, -500)
    
    # 训练和匹配
    matcher = PPFMatcher()
    matcher.train_model(model, sampling_distance_rel=0.025)
    
    matches = matcher.match_scene(
        scene,
        sampling_distance_rel=0.025,
        key_point_fraction=0.1,
        min_score=0.1,
        num_matches=5
    )
    
    print(f"找到 {len(matches)} 个匹配:")
    for i, (pose, score) in enumerate(matches):
        print(f"匹配 {i+1}: 分数 = {score:.4f}")
        
        # 保存变换后的模型
        transformed = transform_pointcloud(model, pose)
        transformed.save(f"result_{i+1}.ply")

if __name__ == "__main__":
    main()
```

### 示例2：使用NumPy数组

```python
#!/usr/bin/env python3

import numpy as np
from ppf_wrapper import PPFMatcher, PointCloud

def create_sample_cube():
    """创建一个立方体点云"""
    points = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                points.append([x, y, z])
    
    # 添加噪声
    points = np.array(points, dtype=np.float32)
    points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
    
    return PointCloud.from_numpy(points)

def main():
    # 创建模型和场景
    model = create_sample_cube()
    model.save("cube_model.ply")
    
    # 创建场景（变换后的模型）
    pose = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 1], 
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    scene = transform_pointcloud(model, pose)
    scene.save("cube_scene.ply")
    
    # 执行匹配
    matcher = PPFMatcher()
    matcher.train_model(model)
    
    matches = matcher.match_scene(scene, num_matches=3)
    
    for i, (found_pose, score) in enumerate(matches):
        print(f"匹配 {i+1}: 分数 = {score:.4f}")
        print("找到的姿态矩阵:")
        print(found_pose)

if __name__ == "__main__":
    main()
```

## 参数调优指南

### 采样距离 (sampling_distance_rel)

- **值范围**: 0.01 - 0.1
- **默认值**: 0.04
- **说明**: 相对于物体直径的采样距离
- **建议**: 
  - 小物体使用较小值 (0.02-0.03)
  - 大物体使用较大值 (0.05-0.08)
  - 噪声大的数据使用较大值

### 关键点比例 (key_point_fraction)

- **值范围**: 0.05 - 0.5
- **默认值**: 0.2
- **说明**: 场景中用作关键点的点比例
- **建议**:
  - 快速匹配使用较小值 (0.05-0.1)
  - 精确匹配使用较大值 (0.2-0.3)

### 最小分数 (min_score)

- **值范围**: 0.0 - 1.0
- **默认值**: 0.2
- **说明**: 返回姿态的最小分数阈值
- **建议**:
  - 噪声大的数据降低阈值 (0.1-0.15)
  - 清洁数据可以提高阈值 (0.3-0.4)

## 性能优化

### 1. 内存管理

```python
# 对大点云进行采样
sampled_model = sample_mesh(model, radius=0.01)
sampled_scene = sample_mesh(scene, radius=0.01)

# 使用采样后的数据进行匹配
matcher.train_model(sampled_model)
matches = matcher.match_scene(sampled_scene)
```

### 2. 模型缓存

```python
# 训练并保存模型
matcher = PPFMatcher()
matcher.train_model(model)
matcher.save_model("cached_model.ppf")

# 后续使用时直接加载
matcher = PPFMatcher()
matcher.load_model("cached_model.ppf")
```

### 3. 并行处理

PPF库自动使用OpenMP进行并行处理，无需额外配置。

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ImportError: No module named 'ppf'
   ```
   **解决**: 确保已正确构建并复制了ppf.so文件

2. **文件未找到**
   ```
   FileNotFoundError: Failed to load PLY file
   ```
   **解决**: 检查文件路径是否正确，确保文件存在

3. **内存不足**
   ```
   MemoryError
   ```
   **解决**: 增加采样距离或使用采样函数减少点数

4. **匹配结果为空**
   ```
   Found 0 matches
   ```
   **解决**: 
   - 降低min_score阈值
   - 增加key_point_fraction
   - 检查数据质量和法向量

### 调试技巧

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查点云质量
print(f"模型点数: {model.num_points}")
print(f"场景点数: {scene.num_points}")
print(f"模型有法向量: {model.has_normals}")
print(f"场景有法向量: {scene.has_normals}")

# 可视化中间结果
model.save("debug_model.ply")
scene.save("debug_scene.ply")
```

## 高级用法

### 自定义法向量计算

```python
# 如果点云没有法向量，可以手动计算
if not model.has_normals:
    # 设置合适的视点
    model.set_view_point(620, 100, 500)
    # 库会自动计算法向量
```

### 批量处理

```python
import glob

def batch_match(model_path, scene_dir, output_dir):
    model = PointCloud.from_file(model_path)
    matcher = PPFMatcher()
    matcher.train_model(model)
    
    scene_files = glob.glob(f"{scene_dir}/*.ply")
    
    for scene_file in scene_files:
        scene = PointCloud.from_file(scene_file)
        matches = matcher.match_scene(scene, num_matches=3)
        
        # 保存结果
        base_name = os.path.basename(scene_file).split('.')[0]
        for i, (pose, score) in enumerate(matches):
            transformed = transform_pointcloud(model, pose)
            transformed.save(f"{output_dir}/{base_name}_match_{i+1}.ply")
```

## 许可证

本项目遵循与原始PPF Surface Match库相同的许可证。

## 贡献

欢迎提交问题和增强请求！