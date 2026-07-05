from typing import Dict, Any, Optional
from pathlib import Path

def create_panorama_data(rawid: str, depth_path: str, hdr_path: str) -> Dict[str, Any]:
    """Create Panorama Data Structure"""
    return {
        "rawid": rawid,
        "type": "panorama",
        "scene": {
            "depth": {
                "path": depth_path,
                "format": "png"
            },
            "hdr": {
                "path": hdr_path,
                "format": "hdr"
            }
        }
    }


def create_multi_view_data(rawid: str, points_data: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Any]:
    """Create Multi-view data Structure"""
    scene = {}
    for pos_id, data in points_data.items():
        point = {}
        # RGB是必需的
        if 'rgb' not in data or not data['rgb']:
            raise ValueError(f"RGB is required for position {pos_id}")
        point['rgb'] = data['rgb']
        
        # 可选字段
        for field in ['depth', 'log', 'normal']:
            point[field] = data.get(field, False)
        
        scene[pos_id] = point
    
    return {
        "rawid": rawid,
        "type": "multi_view_images",
        "scene": scene
    }


def create_video_data(rawid: str, fps: int, duration: float, 
                      frames_data: Dict[str, Dict[str, Optional[str]]],
                      camera_pose_path: Optional[str] = None) -> Dict[str, Any]:
    """Create Video Data Structure"""
    frames = {}
    for frame_id, data in frames_data.items():
        frame = {}
        # RGB是必需的
        if 'rgb' not in data or not data['rgb']:
            raise ValueError(f"RGB is required for frame {frame_id}")
        frame['rgb'] = data['rgb']
        
        # 可选字段
        for field in ['depth', 'normal', 'log']:
            frame[field] = data.get(field, False)
        
        frames[frame_id] = frame
    
    return {
        "rawid": rawid,
        "type": "video_sequence",
        "scene": {
            "sequence": {
                "meta": {
                    "fps": fps,
                    "duration": duration,
                    "camera_pose": camera_pose_path if camera_pose_path else False
                },
                "frames": frames
            }
        }
    }


# 使用示例
if __name__ == "__main__":
    # panorama
    pano = create_panorama_data("rawid1", "path/to/depth_image", "path/to/hdr_image")
    print(pano)
    
    # multi-view
    multi = create_multi_view_data("rawid2", {
        "P1": {"rgb": "path/to/p1_rgb.png", "depth": "path/to/p1_depth.png", 
               "log": "path/to/p1_log.json", "normal": "path/to/p1_normal.png"},
        "P2": {"rgb": "path/to/p2_rgb.png"},
        "P3": {"rgb": "path/to/p3_rgb.png"}
    })
    print(multi)
    
    # video
    video = create_video_data("rawid3", 30, 3.2, {
        "frame_001": {"rgb": "path/to/frame_001_rgb.png", 
                      "depth": "path/to/frame_001_depth.png",
                      "normal": "path/to/frame_001_normal.png",
                      "log": "path/to/frame_001_log.json"},
        "frame_002": {"rgb": "path/to/frame_002_rgb.png",
                      "depth": "path/to/frame_002_depth.png"}
    }, "path/to/camera_pose.json")
    print(video)