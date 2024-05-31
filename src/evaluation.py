import numpy as np

class Evaluator:
    def PSNR(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100

        return 10 * np.log10(255**2 / mse)
    
    def _expand(self, img, d):
        h, w, c = img.shape
        expanded = np.zeros((h * d, w * d, c), dtype=np.uint8)

        for z in range(c):
            for i in range(h):
                for j in range(w):
                    pixel = img[i, j, z]
                    expanded[i*d:(i+1)*d, j*d:(j+1)*d, z] = pixel
        return expanded
    
    def _shrink(self, img, d, method="mean"):
        h, w, c = img.shape
        shrinked = np.zeros((h // d, w // d, c), dtype=np.uint8)

        for z in range(c):
            for i in range(h // d):
                for j in range(w // d):
                    patch = img[i*d:(i+1)*d, j*d:(j+1)*d, z]
                    if method == "mean":
                        pixel = np.mean(patch)
                    elif method == "median":
                        pixel = np.median(patch)
                    shrinked[i, j, z] = pixel
        return shrinked
    
    def ISNR(self, ori_covers, camouflages_chang, camouflages_improved, d, method="expand"):
        n = len(ori_covers)
        for i in range(n):
            if method == "expand":
                expanded = self._expand(ori_covers[i], d)
                isnr = 10 * np.log10(np.mean((camouflages_chang[i] - expanded) ** 2) / np.mean((camouflages_improved[i] - expanded ) ** 2))
            elif method in ["mean", "median"]:
                shrinked_chang = self._shrink(camouflages_chang[i], d, method=method)
                shrinked_improved = self._shrink(camouflages_improved[i], d, method=method)
                isnr = 10 * np.log10(np.mean((shrinked_chang - ori_covers[i]) ** 2) / np.mean((shrinked_improved - ori_covers[i]) ** 2))

        return isnr