
model_data = {
    "model_id": "M000123",
    "model_name": "robot_mech_v2", 
    "source_id": "SRC_20251028_001",
    "components": [
        {
            "id": "C001",
            "name": "UpperArm",
            "parent": "Root",
            "order": 0,
            "md5": "9f7a4a2e...",
            "path": "oss://bucket/M000123/UpperArm.obj"
        },
        {
            "id": "C002",
            "name": "LowerArm",
            "parent": "C001",
            "order": 1,
            "md5": "a2b4cd93...",
            "path": "oss://bucket/M000123/LowerArm.obj"
        }
    ],
    "structure_hash": "d73f0ab91e...",
    "signature": ["Root/C001/UpperArm", "Root/C001/C002/LowerArm"],
    "lineage": {
        "parent": "M000045",
        "pipeline": "spark_dedup_v2"
    }
}

# Stage 1: Structured Object Table
df = spark.createDataFrame([
    ("cup_001", "table_01", (1.2, 0.5, 0.9), (0, 90, 0), (1,1,1), {"type":"cup"}),
    ("table_01", None, (0,0,0), (0,0,0), (1,1,1), {"type":"table"})
], ["object_id","parent_id","position","rotation","scale","attributes"])

# Stage 2: Calculate relative positional relationship
df_rel = df.alias("child").join(df.alias("parent"), ...).select(
    (child.position - parent.position).alias("rel_pos"),
    (child.rotation - parent.rotation).alias("rel_rot")
)

# Stage 3: Generate structure hash
def struct_hash(row):
    base_str = json.dumps({
        "object": row["object_id"],
        "rel_pos": row["rel_pos"],
        "rel_rot": row["rel_rot"]
    }, sort_keys=True)
    return hashlib.md5(base_str.encode()).hexdigest()

