p = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(p, encoding="utf-8").read()

# Add memory cleanup after each patch in the sliding window loop
old = """                    sem_prob = torch.sigmoid(sem_pred[0, 0]).cpu().numpy()
                    emb_vec = emb_pred[0].cpu().numpy()
                    sem_accum[y: y + patch_size, x: x + patch_size] += sem_prob
                    sem_count[y: y + patch_size, x: x + patch_size] += 1.0
                    emb_accum[:, y: y + patch_size, x: x + patch_size] += emb_vec"""

new = """                    sem_prob = torch.sigmoid(sem_pred[0, 0]).cpu().numpy()
                    emb_vec = emb_pred[0].cpu().numpy()
                    del sem_pred, emb_pred, patch_t
                    sem_accum[y: y + patch_size, x: x + patch_size] += sem_prob
                    sem_count[y: y + patch_size, x: x + patch_size] += 1.0
                    emb_accum[:, y: y + patch_size, x: x + patch_size] += emb_vec"""

c = c.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(c)
print("OK")
