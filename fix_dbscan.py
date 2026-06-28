import re
path = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
content = open(path, "r", encoding="utf-8").read()

lines = content.splitlines()
start = None
end = None
for i, line in enumerate(lines):
    if line.strip().startswith("def cluster_embeddings("):
        start = i
    if start is not None and line.strip().startswith("def ") and i > start:
        end = i
        break
if end is None:
    end = len(lines)

print(f"Function from line {start+1} to {end}")

new_func = [
    "def cluster_embeddings(",
    "    embedding: np.ndarray,",
    "    semantic_prob: np.ndarray,",
    "    threshold: float = 0.5,",
    "    eps: float = 0.5,",
    "    min_samples: int = 10,",
    ") -> Tuple[np.ndarray, int]:",
    '    """Connected-component instance separation from centerline probability map."""',
    "",
    "    H, W = semantic_prob.shape",
    "    centerline_mask = (semantic_prob > threshold).astype(np.uint8)",
    "",
    "    if centerline_mask.sum() < min_samples:",
    "        return np.zeros((H, W), dtype=np.int32), 0",
    "",
    "    num_labels, labels = cv2.connectedComponents(centerline_mask, connectivity=8)",
    "",
    "    instance_map = np.zeros((H, W), dtype=np.int32)",
    "    valid_count = 0",
    "    for label_id in range(1, num_labels):",
    "        component_mask = labels == label_id",
    "        if component_mask.sum() >= min_samples:",
    "            valid_count += 1",
    "            instance_map[component_mask] = valid_count",
    "",
    "    return instance_map, valid_count",
    "",
]

new_lines = lines[:start] + new_func + lines[end:]
open(path, "w", encoding="utf-8").write("\n".join(new_lines) + "\n")
print("Done")
