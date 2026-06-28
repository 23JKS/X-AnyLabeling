p = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(p, encoding="utf-8").read()

# 1. Move sklearn import to top-level
c = c.replace(
    "import segmentation_models_pytorch as smp",
    "import segmentation_models_pytorch as smp\nfrom sklearn.cluster import DBSCAN"
)

# 2. Remove lazy import from cluster_embeddings function
c = c.replace(
    '    """DBSCAN clustering on embedding vectors for instance separation."""\n    from sklearn.cluster import DBSCAN\n',
    '    """DBSCAN clustering on embedding vectors for instance separation."""\n'
)

open(p, "w", encoding="utf-8").write(c)
print("OK")
