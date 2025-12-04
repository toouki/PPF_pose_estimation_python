#!/usr/bin/env python3
"""
Example usage of PPF Surface Match Python Wrapper

This example demonstrates how to use the PPF library for 3D object recognition.
"""

import numpy as np
import sys
import os

# Add the python directory to the path to import our wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ppf_wrapper import PPFMatcher, PointCloud, transform_pointcloud
except ImportError as e:
    print(f"Error importing PPF wrapper: {e}")
    print("Please ensure the library is properly built and installed.")
    sys.exit(1)


def main():
    # File paths - adjust these to your actual file locations
    model_file = "../gear.ply"
    scene_file = "../gear_n35.ply"
    
    print("PPF Surface Match Python Example")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print("Please ensure the PLY files are in the correct location.")
        return
    
    if not os.path.exists(scene_file):
        print(f"Scene file not found: {scene_file}")
        print("Please ensure the PLY files are in the correct location.")
        return
    
    try:
        # Create matcher
        matcher = PPFMatcher()
        
        # Load and train model
        print(f"Loading model from: {model_file}")
        model_pc = PointCloud.from_file(model_file)
        print(f"Model has {model_pc.num_points} points")
        print(f"Model has normals: {model_pc.has_normals}")
        
        # Set view point for normal computation if needed
        model_pc.set_view_point(620, 100, 500)
        
        print("Training model...")
        matcher.train_model(model_pc, sampling_distance_rel=0.025)
        print("Model training complete")
        
        # Optionally save the trained model
        model_save_file = "trained_model.ppf"
        matcher.save_model(model_save_file)
        print(f"Trained model saved to: {model_save_file}")
        
        # Load scene
        print(f"\nLoading scene from: {scene_file}")
        scene_pc = PointCloud.from_file(scene_file)
        print(f"Scene has {scene_pc.num_points} points")
        print(f"Scene has normals: {scene_pc.has_normals}")
        
        # Set view point for scene
        scene_pc.set_view_point(-200, -50, -500)
        
        # Perform matching
        print("\nPerforming scene matching...")
        matches = matcher.match_scene(
            scene_pc,
            sampling_distance_rel=0.025,
            key_point_fraction=0.3,
            min_score=0.01,
            num_matches=5
        )
        
        print(f"Found {len(matches)} matches:")
        for i, (pose, score) in enumerate(matches):
            print(f"\nMatch {i+1}:")
            print(f"Score: {score:.4f}")
            print("Pose matrix:")
            print(pose)
            
            # Transform the model using the found pose
            transformed_model = transform_pointcloud(model_pc, pose)
            
            # Save the transformed model
            output_file = f"matched_model_{i}.ply"
            if transformed_model.save(output_file):
                print(f"Transformed model saved to: {output_file}")
            else:
                print(f"Failed to save transformed model to: {output_file}")
        
        print("\nMatching complete!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
        main()