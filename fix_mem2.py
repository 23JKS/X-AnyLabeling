p = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(p, encoding="utf-8").read()

# Add import gc
c = c.replace(
    "import os\nimport cv2\nimport numpy as np\nimport torch",
    "import os\nimport gc\nimport cv2\nimport numpy as np\nimport torch"
)

# Add gc.collect after sliding window loop
old = """        sem_count = np.clip(sem_count, 1e-8, None)
        sem_prob = sem_accum / sem_count
        embedding = emb_accum / sem_count[np.newaxis, :, :]
        return sem_prob[:H_orig, :W_orig], embedding[:, :H_orig, :W_orig]"""

new = """        sem_count = np.clip(sem_count, 1e-8, None)
        sem_prob = sem_accum / sem_count
        embedding = emb_accum / sem_count[np.newaxis, :, :]
        del sem_accum, sem_count, emb_accum
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return sem_prob[:H_orig, :W_orig], embedding[:, :H_orig, :W_orig]"""

c = c.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(c)
print("OK")
