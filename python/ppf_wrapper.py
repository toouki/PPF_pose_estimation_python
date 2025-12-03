"""
PPF Surface Match Python Wrapper

This module provides a Python interface for the PPF (Point Pair Feature) 
surface matching library, which is used for 3D object recognition and pose estimation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import os

try:
    import ppf
except ImportError:
    raise ImportError("Failed to import ppf module. Please ensure the library is properly built and installed.")


class PointCloud:
    """Wrapper class for PPF PointCloud"""
    
    def __init__(self):
        self._pc = ppf.PointCloud()
    
    @classmethod
    def from_file(cls, filename: str) -> 'PointCloud':
        """Load point cloud from PLY file"""
        pc = cls()
        if not ppf.readPLY(filename, pc._pc):
            raise FileNotFoundError(f"Failed to load PLY file: {filename}")
        return pc
    
    @classmethod
    def from_numpy(cls, points: np.ndarray, normals: Optional[np.ndarray] = None) -> 'PointCloud':
        """Create point cloud from numpy arrays"""
        pc = cls()
        
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
        
        # Convert to float32
        points_f32 = points.astype(np.float32)
        
        # Create vector list using numpy arrays
        pc._pc.point = [p for p in points_f32]
        
        if normals is not None:
            if normals.ndim != 2 or normals.shape != points.shape:
                raise ValueError("Normals must have same shape as points")
            
            normals_f32 = normals.astype(np.float32)
            pc._pc.normal = [n for n in normals_f32]
        
        return pc
    
    def to_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert to numpy arrays"""
        points = np.array([[p.x(), p.y(), p.z()] for p in self._pc.point], dtype=np.float32)
        
        normals = None
        if self._pc.hasNormal():
            normals = np.array([[n.x(), n.y(), n.z()] for n in self._pc.normal], dtype=np.float32)
        
        return points, normals
    
    def save(self, filename: str, ascii: bool = False) -> bool:
        """Save point cloud to PLY file"""
        return ppf.writePLY(filename, self._pc, ascii)
    
    @property
    def num_points(self) -> int:
        """Number of points"""
        return self._pc.size()
    
    @property
    def has_normals(self) -> bool:
        """Whether point cloud has normals"""
        return self._pc.hasNormal()
    
    @property
    def bounding_box(self):
        """Get bounding box"""
        return self._pc.box
    
    def set_view_point(self, x: float, y: float, z: float):
        """Set view point for normal computation"""
        import numpy as np
        self._pc.viewPoint = np.array([x, y, z], dtype=np.float32)
    
    def _get_internal(self):
        """Get internal PPF PointCloud object"""
        return self._pc


class PPFMatcher:
    """Main class for PPF surface matching"""
    
    def __init__(self):
        self._detector = ppf.Detector()
        self._model_trained = False
        self._model_diameter = 0.0
    
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
        # Load model if path provided
        if isinstance(model, str):
            model_pc = PointCloud.from_file(model)
        else:
            model_pc = model
        
        # Create training parameters
        train_param = ppf.TrainParam(
            featDistanceStepRel=feat_distance_step_rel,
            featAngleResolution=feat_angle_resolution,
            poseRefRelSamplingDistance=pose_ref_rel_sampling_distance,
            knnNormal=knn_normal,
            smoothNormal=smooth_normal
        )
        
        # Train model
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
        if not self._model_trained:
            raise RuntimeError("Model must be trained before matching")
        
        # Load scene if path provided
        if isinstance(scene, str):
            scene_pc = PointCloud.from_file(scene)
        else:
            scene_pc = scene
        
        # Create matching parameters
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
        
        # Perform matching
        poses = []
        scores = []
        self._detector.matchScene(
            scene_pc._get_internal(), 
            poses, 
            scores, 
            sampling_distance_rel,
            key_point_fraction,
            min_score,
            match_param
        )
        
        # Convert results to numpy arrays
        results = []
        for pose, score in zip(poses, scores):
            pose_matrix = np.array(pose).astype(np.float64)
            results.append((pose_matrix, float(score)))
        
        return results
    
    def save_model(self, filename: str) -> None:
        """Save trained model to file"""
        if not self._model_trained:
            raise RuntimeError("No trained model to save")
        self._detector.save(filename)
    
    def load_model(self, filename: str) -> None:
        """Load trained model from file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        self._detector.load(filename)
        self._model_trained = True


def transform_pointcloud(pc: PointCloud, pose: np.ndarray, use_normal: bool = True) -> PointCloud:
    """Transform point cloud using 4x4 pose matrix"""
    if pose.shape != (4, 4):
        raise ValueError("Pose must be 4x4 matrix")
    
    transformed_pc = PointCloud()
    transformed_pc._pc = ppf.transformPointCloud(pc._get_internal(), pose, use_normal)
    return transformed_pc


def sample_mesh(pc: PointCloud, radius: float) -> PointCloud:
    """Sample mesh with given radius"""
    sampled_pc = PointCloud()
    sampled_pc._pc = ppf.sampleMesh(pc._get_internal(), radius)
    return sampled_pc


def compute_bounding_box(pc: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of point cloud"""
    bbox = ppf.computeBoundingBox(pc._get_internal())
    min_point = np.array([bbox.min.x(), bbox.min.y(), bbox.min.z()])
    max_point = np.array([bbox.max.x(), bbox.max.y(), bbox.max.z()])
    return min_point, max_point