p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()

# Remove sklearn import
c = c.replace('\nfrom sklearn.cluster import DBSCAN', '')

# Replace cluster_embeddings with cv2-based clustering
old_fn = '''def cluster_embeddings(
    semantic_prob: np.ndarray,
    embedding: np.ndarray,
    threshold: float = 0.5,
    eps: float = 0.5,
    min_samples: int = 10,
) -> Tuple[np.ndarray, int]:
    """DBSCAN clustering on embedding vectors for instance separation."""
    from sklearn.cluster import DBSCAN
\n
    H, W = semantic_prob.shape
    D = embedding.shape[0]
    centerline_mask = semantic_prob > threshold
\n
    if centerline_mask.sum() < min_samples:
        return np.zeros((H, W), dtype=np.int32), 0
\n
    ys, xs = np.where(centerline_mask)
    embedding_vectors = embedding[:, ys, xs].transpose(1, 0)
\n
    norms = np.linalg.norm(embedding_vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    embedding_vectors = embedding_vectors / norms
\n
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(embedding_vectors)
\n
    instance_map = np.zeros((H, W), dtype=np.int32)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
\n
    for idx, label in enumerate(unique_labels, start=1):
        mask = labels == label
        instance_map[ys[mask], xs[mask]] = idx
\n
    return instance_map, len(unique_labels)'''

new_fn = '''def cluster_embeddings(
    semantic_prob: np.ndarray,
    embedding: np.ndarray,
    threshold: float = 0.5,
    eps: float = 0.5,
    min_samples: int = 10,
) -> Tuple[np.ndarray, int]:
    """Separate instances using connected components on centerline mask.
       Using cv2 instead of sklearn to avoid Windows thread file errors."""
    H, W = semantic_prob.shape
    centerline_mask = semantic_prob > threshold
    centerline_mask = centerline_mask.astype(np.uint8)

    if centerline_mask.sum() < min_samples:
        return np.zeros((H, W), dtype=np.int32), 0

    n_labels, instance_map = cv2.connectedComponents(centerline_mask, connectivity=8)
    instance_map = instance_map.astype(np.int32)
    return instance_map, n_labels - 1  # subtract background'''

c = c.replace(old_fn, new_fn, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
