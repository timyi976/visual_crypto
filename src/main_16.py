import numpy as np
import cv2
import copy
import random
from tqdm import tqdm

STDIMAGES = "../standard_images/"
SEED = 2024

# set seed
random.seed(SEED)
np.random.seed(SEED)

class StandardImages:
    def __init__(self):
        # color
        self.lenna = cv2.imread(STDIMAGES + "color/lenna_color_512.tif", cv2.IMREAD_COLOR)
        self.mandril = cv2.imread(STDIMAGES + "color/mandril_color.tif", cv2.IMREAD_COLOR)
        self.peppers = cv2.imread(STDIMAGES + "color/peppers_color.tif", cv2.IMREAD_COLOR)
        self.airplaneF16 = cv2.imread(STDIMAGES + "color/airplaneF16.tif", cv2.IMREAD_COLOR)

        # gray
        self.cameraman = cv2.imread(STDIMAGES + "gray/cameraman.tif", cv2.IMREAD_GRAYSCALE)
        self.house = cv2.imread(STDIMAGES + "gray/house.tif", cv2.IMREAD_GRAYSCALE)
        self.jetplane = cv2.imread(STDIMAGES + "gray/jetplane.tif", cv2.IMREAD_GRAYSCALE)
        self.lake = cv2.imread(STDIMAGES + "gray/lake.tif", cv2.IMREAD_GRAYSCALE)

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
            
class VisualCipher:
    def __init__(self):
        pass

    def r2img(self, rs):
        values = np.unique(rs)
        # split 255 into len(values) parts
        split = 255 // len(values)
        rs_img = np.zeros(rs.shape, dtype=np.uint8)
        for i, value in enumerate(values):
            rs_img[rs == value] = split * i

        return rs_img
    
    def img2r(self, rs_img):
        values = np.unique(rs_img)
        # split 255 into len(values) parts
        split = 255 // len(values)
        rs = np.zeros(rs_img.shape, dtype=np.uint8)
        for i, value in enumerate(values):
            rs[rs_img == value] = i + 1

        return rs
    
    def interfold_k(self, k1, k2, r, unpack=True):
        if unpack:
            k1_bits = np.unpackbits(np.array([k1], dtype=np.uint8))
            k2_bits = np.unpackbits(np.array([k2], dtype=np.uint8))
        else:
            k1_bits = np.array(k1)
            k2_bits = np.array(k2)

        # first r bits from k1, next r bits from k2, next 8-r bits from k1, last 8-r bits from k2
        k_bits = np.concatenate((k1_bits[:r], k2_bits[:r], k1_bits[r:], k2_bits[r:]))

        return k_bits
    
    def uninterfold_k(self, k_bits, r):
        k1_bits = np.concatenate((k_bits[:r], k_bits[2*r:2*r+r]))
        k2_bits = np.concatenate((k_bits[r:2*r], k_bits[2*r+r:]))

        k1 = np.packbits(k1_bits)[0]
        k2 = np.packbits(k2_bits)[0]

        return k1, k2
    
    def construct_S_216(self, k1, k2, r, unpack=True):
        # r must between 1 and 8
        assert 1 <= r <= 8, "r must satisfy 1 <= r <= 8"
        # r = r - 1

        S = np.full((2, 16), 3, dtype=np.uint8)

        num_ones = 0

        k_bits = self.interfold_k(k1, k2, r, unpack=unpack)

        num_ones = 0
        for i in range(16):
            if k_bits[i] == 1:
                num_ones += 1

                if num_ones % 2 == 1:
                    S[0, i] = 1
                else:
                    S[0, i] = 0

        for i in range(16):
            if S[0, i] == 3:
                S[0, i] = 1

            S[1, i] = S[0, i] ^ k_bits[i]


        return S
    
    def construct_S_n16(self, k1, k2, r, n):
        assert 1 <= r <= 8, "r must satisfy 1 <= r <= 8"

        S = np.zeros((n, 16), dtype=np.uint8)

        # k_bits = self.interfold_k(k1, k2, r, unpack=True)

        selected = np.random.choice(n, 2, replace=False)

        for i in range(n):
            if i in selected:
                continue
            ones_idx = np.random.choice(16, 8, replace=False)
            S[i, ones_idx] = 1

        S_left = self.construct_S_216(k1, k2, r, unpack=True)

        S[selected[0]] = S_left[0]
        S[selected[1]] = S_left[1]

        return S
    
    def encrypt_n16(self, secret1, secret2, covers):
        h, w, c = secret1.shape
        d = 4
        n = len(covers)

        assert secret1.shape == secret2.shape, "Secret images must have the same shape"
        for cover in covers:
            assert cover.shape == (h, w, c), "All covers must have the same shape as secret images"

        covers = [copy.deepcopy(cover) for cover in covers]

        for cover in covers:
            cover += 1
            cover[cover > 255] = 255

        rs = np.random.randint(1, 9, (h, w), dtype=np.uint8)

        camouflages = []
        for _ in range(n):
            camouflages.append(np.zeros((h*d, w*d, c), dtype=np.uint8))

        pbar = tqdm(total=h*w*c*n, desc="Encrypting")

        for z in range(c):
            for i in range(h):
                for j in range(w):
                    r = rs[i, j]
                    k1 = secret1[i, j, z]
                    k2 = secret2[i, j, z]
                    S = self.construct_S_n16(k1, k2, r, n)
                    for idx in range(n):
                        S_share = S[idx].reshape((d, d))
                        S_share[S_share == 1] = covers[idx][i, j, z]
                        S_share[S_share == 0] = covers[idx][i, j, z] - 1

                        camouflages[idx][i*d:(i+1)*d, j*d:(j+1)*d, z] = S_share

                        pbar.update(1)

        pbar.close()

        return camouflages, rs

    def reconstruct_S_n16(self, patches):
        n = len(patches)
        S = np.zeros((n, 16), dtype=np.uint8)

        patches = [patch.copy().flatten() for patch in patches]

        for i, patch in enumerate(patches):
            one_value = max(patch)
            patch = np.where(patch == one_value, 1, 0)
            S[i] = patch

        return S
    
    def recover_pixels_n16(self, S, r):
        k = np.sum(S, axis=0) % 2
        k1, k2 = self.uninterfold_k(k, r)
        return k1, k2
    
    def decrypt_n16(self, camouflages, rs):
        d = 4
        h, w, c = camouflages[0].shape
        n = len(camouflages)

        secret1 = np.zeros((h//d, w//d, c), dtype=np.uint8)
        secret2 = np.zeros((h//d, w//d, c), dtype=np.uint8)

        pbar = tqdm(total=(h//d)*(h//d)*c, desc="Decrypting")

        for z in range(c):
            for i in range(h//d):
                for j in range(w//d):
                    patches = [camouflage[i*d:(i+1)*d, j*d:(j+1)*d, z] for camouflage in camouflages]
                    S = self.reconstruct_S_n16(patches)
                    r = rs[i, j]
                    k1, k2 = self.recover_pixels_n16(S, r)
                    secret1[i, j, z] = k1
                    secret2[i, j, z] = k2

                    pbar.update(1)

        pbar.close()

        return secret1, secret2

if __name__ == "__main__":
    stdimages = StandardImages()
    secret1 = stdimages.lenna
    secret2 = stdimages.mandril
    covers = [stdimages.peppers, stdimages.airplaneF16]

    vc = VisualCipher()

    # ===== Encrypting =====
    camouflages, rs = vc.encrypt_n16(secret1, secret2, covers)
    # save
    for i, camouflage in enumerate(camouflages):
        cv2.imwrite(f"camouflage_{i}.png", camouflage)
    cv2.imwrite("rs.png", vc.r2img(rs))

    # ===== Decrypting =====
    # read
    camouflages = [cv2.imread(f"camouflage_{i}.png") for i in range(2)]
    rs = vc.img2r(cv2.imread("rs.png", cv2.IMREAD_GRAYSCALE))
    
    secret1_recovered, secret2_recovered = vc.decrypt_n16(camouflages, rs)

    # save
    cv2.imwrite("secret1_recovered.png", secret1_recovered)
    cv2.imwrite("secret2_recovered.png", secret2_recovered)

    # ===== Evaluation =====
    evaluator = Evaluator()
    psnr_secret1 = evaluator.PSNR(secret1, secret1_recovered)
    psnr_secret2 = evaluator.PSNR(secret2, secret2_recovered)
    print(f"PSNR of secret1: {psnr_secret1}")
    print(f"PSNR of secret2: {psnr_secret2}")