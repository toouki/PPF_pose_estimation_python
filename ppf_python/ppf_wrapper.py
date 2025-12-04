"""
PPF Surface Match Python Wrapper

This module provides a Python interface for the PPF (Point Pair Feature) 
surface matching library, which is used for 3D object recognition and pose estimation.
"""

# PPF表面匹配Python包装器
# 
# 该模块为PPF（点对特征）表面匹配库提供Python接口，
# 用于3D物体识别和姿态估计。

import numpy as np
from typing import List, Tuple, Optional, Union
import os

# 尝试导入PPF模块，如果失败则抛出导入错误
try:
    import ppf
except ImportError:
    raise ImportError("Failed to import ppf module. Please ensure the library is properly built and installed.")


class PointCloud:
    """Wrapper class for PPF PointCloud"""
    # PPF点云类的包装器
    
    def __init__(self):
        # 初始化PPF点云对象
        self._pc = ppf.PointCloud()
    
    @classmethod
    def from_file(cls, filename: str) -> 'PointCloud':
        """Load point cloud from PLY file"""
        # 从PLY文件加载点云
        pc = cls()
        if not ppf.readPLY(filename, pc._pc):
            raise FileNotFoundError(f"Failed to load PLY file: {filename}")
        return pc
    
    @classmethod
    def from_numpy(cls, points: np.ndarray, normals: Optional[np.ndarray] = None) -> 'PointCloud':
        """Create point cloud from numpy arrays"""
        # 从numpy数组创建点云
        pc = cls()
        
        # 验证点数组的维度和形状
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
        
        # 转换为float32类型
        points_f32 = points.astype(np.float32)
        
        # 使用numpy数组创建向量列表
        pc._pc.point = [p for p in points_f32]
        
        # 如果提供了法向量，则设置法向量
        if normals is not None:
            # 验证法向量数组的维度和形状
            if normals.ndim != 2 or normals.shape != points.shape:
                raise ValueError("Normals must have same shape as points")
            
            normals_f32 = normals.astype(np.float32)
            pc._pc.normal = [n for n in normals_f32]
        
        return pc
    
    def to_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert to numpy arrays"""
        # 转换为numpy数组
        # 提取点坐标
        points = np.array([[p.x(), p.y(), p.z()] for p in self._pc.point], dtype=np.float32)
        
        normals = None
        # 如果点云包含法向量，则提取法向量
        if self._pc.hasNormal():
            normals = np.array([[n.x(), n.y(), n.z()] for n in self._pc.normal], dtype=np.float32)
        
        return points, normals
    
    def save(self, filename: str, ascii: bool = False) -> bool:
        """Save point cloud to PLY file"""
        # 保存点云到PLY文件
        return ppf.writePLY(filename, self._pc, ascii)
    
    @property
    def num_points(self) -> int:
        """Number of points"""
        # 点的数量
        return self._pc.size()
    
    @property
    def has_normals(self) -> bool:
        """Whether point cloud has normals"""
        # 点云是否包含法向量
        return self._pc.hasNormal()
    
    @property
    def bounding_box(self):
        """Get bounding box"""
        # 获取边界框
        return self._pc.box
    
    def set_view_point(self, x: float, y: float, z: float):
        """Set view point for normal computation"""
        # 设置法向量计算的视点
        import numpy as np
        self._pc.viewPoint = np.array([x, y, z], dtype=np.float32)
    
    def _get_internal(self):
        """Get internal PPF PointCloud object"""
        # 获取内部PPF点云对象
        return self._pc


class PPFMatcher:
    """Main class for PPF surface matching"""
    # PPF表面匹配的主要类
    
    def __init__(self):
        # 初始化PPF检测器
        self._detector = ppf.Detector()
        self._model_trained = False  # 模型是否已训练
        self._model_diameter = 0.0   # 模型直径
    
    def train_model(self, 
                   model: Union[PointCloud, str], 
                   sampling_distance_rel: float = 0.04,
                   feat_distance_step_rel: float = 0.04,
                   feat_angle_resolution: int = 30,
                   pose_ref_rel_sampling_distance: float = 0.01,
                   knn_normal: int = 10,
                   smooth_normal: bool = True) -> None:
        """
        Train PPF model
        
        Args:
            model: PointCloud object or path to PLY file
            sampling_distance_rel: Sampling distance relative to object diameter
            feat_distance_step_rel: Feature distance step relative to object diameter
            feat_angle_resolution: Feature angle resolution
            pose_ref_rel_sampling_distance: Pose refinement sampling distance
            knn_normal: Number of neighbors for normal estimation
            smooth_normal: Whether to smooth normals
        """
        # 训练PPF模型
        #
        # 参数:
        #     model: 点云对象或PLY文件路径
        #     sampling_distance_rel: 相对于物体直径的采样距离
        #     feat_distance_step_rel: 相对于物体直径的特征距离步长
        #     feat_angle_resolution: 特征角度分辨率
        #     pose_ref_rel_sampling_distance: 姿态精化采样距离
        #     knn_normal: 法向量估计的近邻数量
        #     smooth_normal: 是否平滑法向量
        
        # 如果提供了文件路径，则加载模型
        if isinstance(model, str):
            model_pc = PointCloud.from_file(model)
        else:
            model_pc = model
        
        # 创建训练参数
        train_param = ppf.TrainParam(
            featDistanceStepRel=feat_distance_step_rel,
            featAngleResolution=feat_angle_resolution,
            poseRefRelSamplingDistance=pose_ref_rel_sampling_distance,
            knnNormal=knn_normal,
            smoothNormal=smooth_normal
        )
        
        # 训练模型
        self._detector.trainModel(model_pc._get_internal(), sampling_distance_rel, train_param)
        self._model_trained = True
        self._model_diameter = model_pc.bounding_box.diameter()
    
    def match_scene(self,
                   scene: Union[PointCloud, str],
                   sampling_distance_rel: float = 0.04,
                   key_point_fraction: float = 0.2,
                   min_score: float = 0.2,
                   num_matches: int = 1,
                   knn_normal: int = 10,
                   smooth_normal: bool = True,
                   invert_normal: bool = False,
                   max_overlap_dist_rel: float = 0.5,
                   sparse_pose_refinement: bool = True,
                   dense_pose_refinement: bool = True,
                   pose_ref_num_steps: int = 5,
                   pose_ref_dist_threshold_rel: float = 0.1,
                   pose_ref_scoring_dist_rel: float = 0.01) -> List[Tuple[np.ndarray, float]]:
        """
        Match trained model in scene
        
        Args:
            scene: PointCloud object or path to PLY file
            sampling_distance_rel: Scene sampling distance relative to model diameter
            key_point_fraction: Fraction of scene points used as key points
            min_score: Minimum score for returned poses
            num_matches: Maximum number of matches to return
            knn_normal: Number of neighbors for normal estimation
            smooth_normal: Whether to smooth normals
            invert_normal: Whether to invert scene normals
            max_overlap_dist_rel: Maximum overlap distance relative to model diameter
            sparse_pose_refinement: Whether to use sparse pose refinement
            dense_pose_refinement: Whether to use dense pose refinement
            pose_ref_num_steps: Number of pose refinement steps
            pose_ref_dist_threshold_rel: Distance threshold for pose refinement
            pose_ref_scoring_dist_rel: Distance threshold for scoring
            
        Returns:
            List of tuples (pose_matrix, score) where pose_matrix is 4x4 transformation matrix
        """
        # 在场景中匹配已训练的模型
        #
        # 参数:
        #     scene: 点云对象或PLY文件路径
        #     sampling_distance_rel: 相对于模型直径的场景采样距离
        #     key_point_fraction: 用作关键点的场景点比例
        #     min_score: 返回姿态的最小分数
        #     num_matches: 返回的最大匹配数量
        #     knn_normal: 法向量估计的近邻数量
        #     smooth_normal: 是否平滑法向量
        #     invert_normal: 是否反转场景法向量
        #     max_overlap_dist_rel: 相对于模型直径的最大重叠距离
        #     sparse_pose_refinement: 是否使用稀疏姿态精化
        #     dense_pose_refinement: 是否使用密集姿态精化
        #     pose_ref_num_steps: 姿态精化步数
        #     pose_ref_dist_threshold_rel: 姿态精化的距离阈值
        #     pose_ref_scoring_dist_rel: 评分的距离阈值
        #     
        # 返回:
        #     元组列表 (pose_matrix, score)，其中pose_matrix是4x4变换矩阵
        
        # 检查模型是否已训练
        if not self._model_trained:
            raise RuntimeError("Model must be trained before matching")
        
        # 如果提供了文件路径，则加载场景
        if isinstance(scene, str):
            scene_pc = PointCloud.from_file(scene)
        else:
            scene_pc = scene
        
        # 创建匹配参数
        match_param = ppf.MatchParam(
            numMatches=num_matches,
            knnNormal=knn_normal,
            smoothNormal=smooth_normal,
            invertNormal=invert_normal,
            maxOverlapDistRel=max_overlap_dist_rel,
            sparsePoseRefinement=sparse_pose_refinement,
            densePoseRefinement=dense_pose_refinement,
            poseRefNumSteps=pose_ref_num_steps,
            poseRefDistThresholdRel=pose_ref_dist_threshold_rel,
            poseRefScoringDistRel=pose_ref_scoring_dist_rel
        )
        
        # 执行匹配
        poses, scores = self._detector.matchScene(
            scene_pc._get_internal(), 
            sampling_distance_rel,
            key_point_fraction,
            min_score,
            match_param
        )
        
        # 将结果转换为numpy数组
        results = []
        for pose, score in zip(poses, scores):
            pose_matrix = np.array(pose).astype(np.float64)
            results.append((pose_matrix, float(score)))
        
        return results
    
    def save_model(self, filename: str) -> None:
        """Save trained model to file"""
        # 保存训练好的模型到文件
        if not self._model_trained:
            raise RuntimeError("No trained model to save")
        self._detector.save(filename)
    
    def load_model(self, filename: str) -> None:
        """Load trained model from file"""
        # 从文件加载训练好的模型
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        self._detector.load(filename)
        self._model_trained = True


def transform_pointcloud(pc: PointCloud, pose: np.ndarray, use_normal: bool = True) -> PointCloud:
    """Transform point cloud using 4x4 pose matrix"""
    # 使用4x4姿态矩阵变换点云
    if pose.shape != (4, 4):
        raise ValueError("Pose must be 4x4 matrix")
    
    transformed_pc = PointCloud()
    transformed_pc._pc = ppf.transformPointCloud(pc._get_internal(), pose, use_normal)
    return transformed_pc


def sample_mesh(pc: PointCloud, radius: float) -> PointCloud:
    """Sample mesh with given radius"""
    # 使用给定半径对网格进行采样
    sampled_pc = PointCloud()
    sampled_pc._pc = ppf.sampleMesh(pc._get_internal(), radius)
    return sampled_pc


def compute_bounding_box(pc: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of point cloud"""
    # 计算点云的边界框
    bbox = ppf.computeBoundingBox(pc._get_internal())
    min_point = np.array([bbox.min.x(), bbox.min.y(), bbox.min.z()])
    max_point = np.array([bbox.max.x(), bbox.max.y(), bbox.max.z()])
    return min_point, max_point