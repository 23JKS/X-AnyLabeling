p = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(p, encoding="utf-8").read()

# Add deque to top-level imports
c = c.replace(
    "from typing import Optional, Tuple",
    "from typing import Optional, Tuple\nfrom collections import deque"
)

# Remove from _bfs_skeleton_path
c = c.replace(
    '    @staticmethod\n    def _bfs_skeleton_path(skeleton: np.ndarray, start: tuple, end: tuple):\n        from collections import deque\n',
    '    @staticmethod\n    def _bfs_skeleton_path(skeleton: np.ndarray, start: tuple, end: tuple):\n'
)

open(p, "w", encoding="utf-8").write(c)
print("OK")
