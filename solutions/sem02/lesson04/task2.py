import numpy as np
from numpy.lib.stride_tricks import as_strided


def get_dominant_color_info(
    image: np.ndarray[np.uint8],
    threshold: int = 5,
) -> tuple[np.uint8, float]:

    if threshold < 1:
        raise ValueError("threshold must be positive")

    threshold -= 1
    cnt = np.bincount(image.getfield(np.uint8).ravel(), minlength=256)
    ans = np.zeros(2 * (threshold + 1) + cnt.shape[0])
    ans[threshold + 1 : ans.shape[0] - threshold - 1] = cnt
    ans = np.cumsum(ans)
    ans = (
        ans[2 * threshold + 1 : ans.shape[0] - 1] - ans[: ans.shape[0] - 1 - 2 * threshold - 1]
    ) * (cnt != 0)

    index = np.arange(256, dtype=np.uint8)
    mx = np.max(ans)
    ans1 = (mx == ans) * index + (mx != ans) * -1
    ans2 = np.max(ans * (ans1 != -1)) / np.sum(cnt)
    ans1 = ans1[ans1 != -1]

    return ans1[0].astype(np.uint8), ans2
