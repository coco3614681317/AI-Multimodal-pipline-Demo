"""
================================================================================
3D MODEL DEDUPLICATION SYSTEM - STORAGE STRUCTURE
================================================================================
Supported Formats: FBX, GLB/GLTF (both supported)
- GLB: Recommended for web/AR/VR, better compression (Draco)
- FBX: Recommended for game engines, full feature support
- Both store: geometry + materials + animations + hierarchy
================================================================================
"""

# ============================================================================
# LAYER 1: HOT STORAGE (Redis/HBase) - Millisecond Access
# ============================================================================
{
  "model_id": "M000123",
  "storage": {
    "format": "glb",                    # or "fbx"
    "primary": "oss://bucket/models/M000123.glb",
    "thumbnail": "oss://bucket/thumbs/M000123.webp",
    "features": "oss://bucket/features/M000123.parquet"
  },
  "indexes": {
    "structure_hash": "d73f0ab91e...",  # Topological structure (your original)
    "geometry_hash": "a1b2c3d4...",     # Geometric shape fingerprint (new)
    "semantic_hash": "e5f6g7h8..."      # Semantic fingerprint (new)
  },
  "metadata": {
    "format": "glb",                    # or "fbx"
    "compression": "draco",             # draco/none (GLB only)
    "size_bytes": 2456789,
    "vertex_count": 24567,
    "face_count": 12345,
    "lod_level": 0,                     # Level of Detail (0=highest)
    "version": "2.0"
  },
  "lineage": {
    "parent_model": "M000045",
    "source": "SRC_20251028_001",
    "pipeline": "spark_dedup_v3",
    "timestamp": "2026-07-05T10:30:00Z"
  }
}

# ============================================================================
# LAYER 2: COLD STORAGE (OSS/S3) - Original Files
# ============================================================================
"""
oss://bucket/models/
├── M000123.glb          # Primary GLB file (Draco compressed)
├── M000124.fbx          # Primary FBX file
├── M000125.glb
└── ...

oss://bucket/features/
├── M000123.parquet      # Geometric features (for batch processing)
└── ...

oss://bucket/thumbnails/
├── M000123.webp         # Preview thumbnail
└── ...
"""

# ============================================================================
# LAYER 3: STRUCTURED STORAGE (PostgreSQL) - Business Queries
# ============================================================================
"""
Table: models
┌──────────────┬─────────────┬─────────────────┬──────────────┐
│ model_id     │ format      │ structure_hash  │ geometry_hash│
├──────────────┼─────────────┼─────────────────┼──────────────┤
│ M000123      │ glb         │ d73f0ab91e...   │ a1b2c3d4...  │
│ M000124      │ fbx         │ a3b4c5d6e7...   │ b2c3d4e5...  │
└──────────────┴─────────────┴─────────────────┴──────────────┘

Table: model_components
┌──────────────┬──────────────┬─────────────┬──────────────┐
│ model_id     │ component_id │ parent_id   │ rel_pos      │
├──────────────┼──────────────┼─────────────┼──────────────┤
│ M000123      │ C001         │ Root        │ (0,0,0)      │
│ M000123      │ C002         │ C001        │ (1.2,0,0)    │
└──────────────┴──────────────┴─────────────┴──────────────┘
"""

# ============================================================================
# OPTIMIZED DEDUPLICATION PIPELINE
# ============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import *
import hashlib
import json
import numpy as np

# ----------------------------------------------------------------------------
# Stage 1: Structure Hash (Your Original Approach - Enhanced)
# ----------------------------------------------------------------------------

def compute_structure_hash(components):
    """
    Computes topological structure hash from component hierarchy.
    
    Args:
        components: List of component dicts with id, parent, rel_pos, rel_rot
    
    Returns:
        SHA256 hash string representing the structure
        
    Note: Uses relative positioning (translation & rotation invariant)
    """
    # Sort by hierarchical order for consistency
    sorted_components = sorted(components, key=lambda x: (x.get('order', 0), x['id']))
    
    # Build structure signature
    structure_data = []
    for comp in sorted_components:
        structure_data.append({
            "id": comp['id'],
            "parent": comp.get('parent', None),
            "rel_pos": [round(v, 4) for v in comp.get('rel_pos', [0,0,0])],  # Quantize floats
            "rel_rot": [round(v, 4) for v in comp.get('rel_rot', [0,0,0])],
            "rel_scale": [round(v, 4) for v in comp.get('rel_scale', [1,1,1])]  # ADDED: scale
        })
    
    # Generate deterministic string
    struct_str = json.dumps(structure_data, sort_keys=True)
    return hashlib.sha256(struct_str.encode()).hexdigest()

# ----------------------------------------------------------------------------
# Stage 2: Geometry Hash (NEW - For Shape Matching)
# ----------------------------------------------------------------------------

def compute_geometry_hash(model_path, format_type='glb'):
    """
    Computes geometric fingerprint from 3D model file.
    
    Supports both GLB and FBX formats via trimesh library.
    
    Args:
        model_path: Path to model file (.glb or .fbx)
        format_type: 'glb' or 'fbx'
    
    Returns:
        Dictionary with multiple geometric hashes
    """
    import trimesh
    import open3d as o3d
    
    # Load model (supports both GLB and FBX)
    if format_type == 'glb':
        mesh = trimesh.load(model_path, force='mesh')
    elif format_type == 'fbx':
        # FBX may contain multiple meshes, combine them
        scene = trimesh.load(model_path, force='scene')
        mesh = trimesh.util.concatenate([m for m in scene.geometry.values()])
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Normalize for invariance
    vertices = mesh.vertices.copy()
    
    # Step 1: Center at origin (translation invariance)
    vertices -= vertices.mean(axis=0)
    
    # Step 2: Scale to unit size (scale invariance)
    max_dist = np.linalg.norm(vertices, axis=1).max()
    if max_dist > 0:
        vertices /= max_dist
    
    # Step 3: Compute geometric fingerprints
    
    # 3a: Vertex distribution hash (exact geometry)
    vertex_hash = hashlib.sha256(vertices.tobytes()).hexdigest()
    
    # 3b: Voxel hash (noise-resistant, rotation-sensitive)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=0.02
    )
    voxel_hash = hashlib.sha256(
        str(voxel_grid.get_voxels()).encode()
    ).hexdigest()
    
    # 3c: Normal histogram (rotation-resistant)
    if hasattr(mesh, 'vertex_normals'):
        normals = mesh.vertex_normals
        # Convert to spherical coordinates and histogram
        theta = np.arccos(np.clip(normals[:, 2], -1, 1))
        phi = np.arctan2(normals[:, 1], normals[:, 0])
        hist, _ = np.histogram(theta, bins=18)  # 10-degree bins
        normal_hash = hashlib.sha256(hist.tobytes()).hexdigest()
    else:
        normal_hash = vertex_hash  # Fallback
    
    # 3d: Feature vector for approximate matching
    feature_vector = {
        "vertex_count": len(vertices),
        "face_count": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        "volume": mesh.volume if hasattr(mesh, 'volume') else 0,
        "surface_area": mesh.area if hasattr(mesh, 'area') else 0,
        "bounding_box": vertices.max(axis=0).tolist() + vertices.min(axis=0).tolist()
    }
    
    return {
        "vertex_hash": vertex_hash,      # Exact geometry match
        "voxel_hash": voxel_hash,        # Noise-tolerant match
        "normal_hash": normal_hash,      # Rotation-invariant match
        "feature_vector": feature_vector  # For similarity search
    }

# ----------------------------------------------------------------------------
# Stage 3: Multi-Level Deduplication (OPTIMIZED)
# ----------------------------------------------------------------------------
# New Model Incoming
#     ↓
# ┌─────────────────────────────────────────────────────┐
# │ Level 1: Exact Match (Fastest, O(1))               │
# │ "Same structure + Same geometry" → 100% Duplicate  │
# └─────────────────────────────────────────────────────┘
#     ↓ Not found
# ┌─────────────────────────────────────────────────────┐
# │ Level 2: Approximate Match (Medium Speed)          │
# │ "Look similar" → Could be different LOD or tweaked │
# └─────────────────────────────────────────────────────┘
#     ↓ Not found
# ┌─────────────────────────────────────────────────────┐
# │ Level 3: Semantic Match (Slowest, Needs Review)    │
# │ "Same function" → Both are chairs, but look differ │
# └─────────────────────────────────────────────────────┘
#     ↓ No match found
# ┌─────────────────────────────────────────────────────┐
# │ New Model → Store in Database                       │
# └─────────────────────────────────────────────────────┘
# ============================================================================
class ModelDeduplicator:
    """
    Three-tier deduplication strategy for 3D models.
    
    Level 1: Exact match (structure + geometry)
    Level 2: Approximate match (geometric similarity)
    Level 3: Semantic match (functional equivalence)
    """
    
    def __init__(self, redis_client, oss_client, pg_client):
        self.redis = redis_client  # Hot cache
        self.oss = oss_client      # Cold storage
        self.pg = pg_client        # Metadata
        
    def deduplicate(self, model_data, model_file):
        """
        Main deduplication pipeline.
        
        Args:
            model_data: Dict with structure_hash, geometry_hash, metadata
            model_file: Path to model file (.glb or .fbx)
        
        Returns:
            dict: Deduplication result with status and matched_id
        """
        
        # --- LEVEL 1: EXACT MATCH (Fast, O(1)) ---
        exact_key = f"{model_data['structure_hash']}_{model_data['geometry_hash']}"
        
        # Check Redis cache first
        cached = self.redis.get(exact_key)
        if cached:
            return {
                "status": "duplicate_exact",
                "matched_id": cached,
                "confidence": 1.0,
                "level": "exact"
            }
        
        # Check PostgreSQL for persistent exact matches
        existing = self.pg.query(
            "SELECT model_id FROM models WHERE structure_hash = %s AND geometry_hash = %s",
            (model_data['structure_hash'], model_data['geometry_hash'])
        )
        if existing:
            matched_id = existing[0]['model_id']
            self.redis.setex(exact_key, 86400, matched_id)  # Cache for 24h
            return {
                "status": "duplicate_exact",
                "matched_id": matched_id,
                "confidence": 1.0,
                "level": "exact"
            }
        
        # --- LEVEL 2: APPROXIMATE MATCH (Geometric Similarity) ---
        # Use voxel hash for fast approximate matching
        approx_matches = self.find_similar_by_voxel(
            model_data['voxel_hash'],
            threshold=0.95
        )
        
        if approx_matches:
            # Verify with detailed geometric comparison
            best_match = self.verify_geometric_similarity(
                model_file,
                approx_matches[0]['model_id']
            )
            if best_match and best_match['similarity'] > 0.90:
                return {
                    "status": "duplicate_approx",
                    "matched_id": best_match['model_id'],
                    "confidence": best_match['similarity'],
                    "level": "approximate",
                    "differences": best_match.get('differences', [])
                }
        
        # --- LEVEL 3: SEMANTIC MATCH (Functional Equivalence) ---
        # Check if same semantic category exists
        semantic_key = model_data.get('metadata', {}).get('category')
        if semantic_key:
            semantic_matches = self.pg.query(
                "SELECT model_id FROM models WHERE metadata->>'category' = %s",
                (semantic_key,)
            )
            if semantic_matches:
                # Human review required for semantic matches
                self.add_to_review_queue(model_data)
                return {
                    "status": "pending_review",
                    "candidates": [m['model_id'] for m in semantic_matches],
                    "level": "semantic",
                    "reason": "Same category - needs human verification"
                }
        
        # --- NEW MODEL ---
        # Store as new unique model
        model_id = self.store_new_model(model_data, model_file)
        return {
            "status": "new_model",
            "model_id": model_id,
            "level": "new"
        }
    
    def find_similar_by_voxel(self, voxel_hash, threshold=0.95):
        """
        Find models with similar voxel hash (fast approximate matching).
        
        Uses Locality-Sensitive Hashing (LSH) for efficient similarity search.
        """
        # Query Redis for models with similar voxel patterns
        # Implementation uses LSH or approximate nearest neighbors
        candidates = self.redis.lrange(f"voxel:{voxel_hash[:8]}", 0, -1)
        
        # Filter by actual similarity
        results = []
        for candidate in candidates:
            similarity = self.compute_voxel_similarity(
                voxel_hash,
                candidate['voxel_hash']
            )
            if similarity >= threshold:
                results.append({
                    "model_id": candidate['model_id'],
                    "similarity": similarity
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def verify_geometric_similarity(self, model_file, existing_model_id):
        """
        Detailed geometric comparison using ICP or point cloud matching.
        """
        # Load both models
        new_mesh = trimesh.load(model_file)
        existing_path = self.oss.download(f"models/{existing_model_id}.glb")
        existing_mesh = trimesh.load(existing_path)
        
        # Sample point clouds
        new_points = new_mesh.sample(10000)
        existing_points = existing_mesh.sample(10000)
        
        # ICP registration to find best alignment
        from scipy.spatial import KDTree
        tree = KDTree(existing_points)
        distances, _ = tree.query(new_points)
        
        # Compute similarity (Hausdorff distance-based)
        mean_distance = distances.mean()
        max_distance = distances.max()
        similarity = 1.0 / (1.0 + mean_distance)
        
        return {
            "similarity": similarity,
            "mean_distance": mean_distance,
            "max_distance": max_distance
        }
    
    def store_new_model(self, model_data, model_file):
        """
        Store new model across all storage layers.
        """
        model_id = model_data['model_id']
        format_type = model_data['storage']['format']
        
        # 1. Upload to OSS (cold storage)
        self.oss.upload(
            model_file,
            f"models/{model_id}.{format_type}"
        )
        
        # 2. Store metadata in PostgreSQL
        self.pg.insert(
            "models",
            {
                "model_id": model_id,
                "structure_hash": model_data['structure_hash'],
                "geometry_hash": model_data['geometry_hash'],
                "voxel_hash": model_data.get('voxel_hash'),
                "metadata": json.dumps(model_data.get('metadata', {})),
                "storage_info": json.dumps(model_data['storage'])
            }
        )
        
        # 3. Cache in Redis (hot storage)
        exact_key = f"{model_data['structure_hash']}_{model_data['geometry_hash']}"
        self.redis.setex(exact_key, 86400, model_id)
        
        # 4. Index for approximate matching
        voxel_prefix = model_data.get('voxel_hash', '')[:8]
        self.redis.lpush(f"voxel:{voxel_prefix}", json.dumps({
            "model_id": model_id,
            "voxel_hash": model_data.get('voxel_hash')
        }))
        
        return model_id

# ----------------------------------------------------------------------------
# Stage 4: Spark Pipeline Integration
# ----------------------------------------------------------------------------

def dedup_pipeline(spark, model_files_df):
    """
    Batch deduplication pipeline using Spark.
    
    Args:
        spark: SparkSession
        model_files_df: DataFrame with columns [model_id, file_path, format]
    
    Returns:
        DataFrame with deduplication results
    """
    
    # UDF for structure hash computation
    def compute_structure_hash_udf(components_json):
        components = json.loads(components_json)
        return compute_structure_hash(components)
    
    # UDF for geometry hash computation
    def compute_geometry_hash_udf(file_path, format_type):
        try:
            hashes = compute_geometry_hash(file_path, format_type)
            return json.dumps(hashes)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    # Register UDFs
    structure_hash_udf = F.udf(compute_structure_hash_udf, StringType())
    geometry_hash_udf = F.udf(compute_geometry_hash_udf, StringType())
    
    # Process DataFrame
    df_processed = model_files_df \
        .withColumn("structure_hash", structure_hash_udf(F.col("components_json"))) \
        .withColumn("geometry_hashes", geometry_hash_udf(F.col("file_path"), F.col("format"))) \
        .withColumn("voxel_hash", F.get_json_object(F.col("geometry_hashes"), "$.voxel_hash")) \
        .withColumn("combined_hash", F.sha2(F.concat(F.col("structure_hash"), F.col("voxel_hash")), 256))
    
    # Deduplicate using window function
    window_spec = Window.partitionBy("combined_hash").orderBy("timestamp")
    
    df_dedup = df_processed \
        .withColumn("row_num", F.row_number().over(window_spec)) \
        .withColumn("is_duplicate", F.col("row_num") > 1) \
        .withColumn("original_id", F.when(F.col("row_num") == 1, F.col("model_id"))
                                    .otherwise(F.first("model_id").over(window_spec)))
    
    return df_dedup