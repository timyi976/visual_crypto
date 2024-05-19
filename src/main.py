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

    def construct_S_29(self, k, r, unpack=True):
        # Use 3 to represent null elements
        S = np.full((2, 9), 3, dtype=np.uint8)
        
        num_ones = 0

        # convert k into bits
        if unpack:
            k_bits = np.unpackbits(np.array([k], dtype=np.uint8))
        else:
            k_bits = k

        for i in range(8):
            if i < r-1 and k_bits[i] == 1:
                num_ones += 1
                if num_ones % 2 == 0:
                    S[0, i] = 0
                else:
                    S[0, i] = 1
            elif i >= r-1 and k_bits[i] == 1:
                num_ones += 1
                j = i + 1
                if num_ones % 2 == 1:
                    S[0, j] = 1
                else:
                    S[0, j] = 0

        if num_ones % 2 == 1:
            S[0, r-1] = 0

        # Randomly assign the rest null elements in row 1
        # (To simplify the problem, we assign 1 to the null in row 1)
        for i in range(9):
            if S[0,i] == 3:
                S[0,i] = 1

        #Compute row 2
        j = -1
        for i in range(9):
            if i != r-1:
                j += 1
                S[1,i] = S[0,i] ^ k_bits[j]
            else:
                S[1,i] = S[0,i] ^ (num_ones % 2)
                
        return S

    def construct_S_n9(self, k, r, n):
        # r must be between 1 and m
        assert 1 <= r <= 9, "r must satisfy 1 <= r <= 9"
        
        S = np.zeros((n, 9), dtype=np.uint8)

        k_bits = np.unpackbits(np.array([k], dtype=np.uint8))

        selected = np.random.choice(n, 2, replace=False)
        
        for i in range(n):
            if i in selected:
                continue
            ones_indices = np.random.choice(9, 5, replace=False)
            S[i, ones_indices] = 1

        # Compute t
        t = np.zeros(9, dtype=np.uint8)

        for i in range(9):
            if i < r - 1:
                t[i] = k_bits[i]
            elif i == r - 1:
                t[i] = np.random.randint(0, 2)
            else:
                t[i] = k_bits[i - 1]

        t_prime = np.bitwise_xor(t, np.sum(S, axis=0) % 2)

        k_prime = np.delete(t_prime, r - 1)
        S_left = self.construct_S_29(k_prime, r, unpack=False)

        S[selected[0]] = S_left[0]
        S[selected[1]] = S_left[1]

        return S
    
    def encrypt_n9(self, secret, covers, improve=True):
        h, w, c = secret.shape
        d = 3
        n = len(covers)

        # All cover images must share the same shape
        for cover in covers:
            assert cover.shape == secret.shape, "All cover images must share the same shape"

        # Make deep copies of covers to ensure original covers are not modified
        covers = [copy.deepcopy(cover) for cover in covers]

        if improve:
            # Add 1 to all cover images
            for cover in covers:
                cover += 1
                # If exceeding 255, set to 255
                cover[cover > 255] = 255
        else:
            # Ensure distinct value separation for non-improved version
            for cover in covers:
                cover[cover < 128] += 1
                cover[cover >= 128] -= 1

        # Generate random numbers in shape of (h, w)
        rs = np.random.randint(1, 10, (h, w))

        # Camouflages
        camouflages = []
        for _ in range(n):
            camouflages.append(np.zeros((h * d, w * d, c), dtype=np.uint8))

        pbar = tqdm(total=h * w * c * n, desc="Encrypting")

        for z in range(c):
            for i in range(h):
                for j in range(w):
                    r = rs[i, j]
                    k = secret[i, j, z]
                    S = self.construct_S_n9(k, r, n)
                    for idx in range(n):
                        S_share = S[idx].reshape(d, d)
                        S_share[S_share == 1] = covers[idx][i, j, z]
                        if improve:
                            S_share[S_share == 0] = covers[idx][i, j, z] - 1
                        else:
                            S_share[S_share == 0] = 0

                        # Put S_share into camouflages
                        camouflages[idx][i * d:(i + 1) * d, j * d:(j + 1) * d, z] = S_share

                        pbar.update(1)

        pbar.close()

        return camouflages, rs
    
    def reconstruct_S_n9(self, patches):
        n = len(patches)
        S = np.zeros((n, 9), dtype=np.int8)

        patches = [patch.copy().flatten() for patch in patches]

        for i, patch in enumerate(patches):
            one_value = max(patch)
            patch = np.where(patch == one_value, 1, 0)
            S[i] = patch

        return S

    def recover_pixel_n9(self, S, r):
        t = np.sum(S, axis=0) % 2

        k = 0
        for j in range(9):
            if j == r-1:
                continue
            elif j < r-1:
                k += t[j] * (2 ** (7-j))
            elif j > r-1:
                k += t[j] * (2 ** (7-j+1))
        return k
    
    def decrypt_n9(self, camouflages, rs):
        d = 3
        h, w, c = camouflages[0].shape
        n = len(camouflages)
        
        secret = np.zeros((h // d, w // d, c), dtype=np.uint8)

        pbar = tqdm(total=(h // d) * (w // d) * c, desc="Decrypting")

        for z in range(c):
            for i in range(h // d):
                for j in range(w // d):
                    patches = [camouflage[i * d:(i + 1) * d, j * d:(j + 1) * d, z] for camouflage in camouflages]
                    S = self.reconstruct_S_n9(patches)
                    r = rs[i, j]
                    k = self.recover_pixel_n9(S, r)
                    secret[i, j, z] = k
                    pbar.update(1)

        pbar.close()

        return secret
    
if __name__ == "__main__":
    # ===== Prepare standard images =====
    std_images = StandardImages()
    secret = std_images.lenna
    covers = [std_images.mandril, std_images.peppers, std_images.airplaneF16]

    vc = VisualCipher()
    
    # ===== Encrypt =====
    camouflages_ori, rs_ori = vc.encrypt_n9(secret, covers, improve=False)
    camouflages, rs = vc.encrypt_n9(secret, covers, improve=True)
    # save
    for idx in range(len(camouflages)):
        cv2.imwrite(f"camouflage_{idx}.png", camouflages[idx])
        cv2.imwrite(f"camouflage_ori_{idx}.png", camouflages_ori[idx])
    cv2.imwrite("rs.png", vc.r2img(rs))
    cv2.imwrite("rs_ori.png", vc.r2img(rs_ori))


    # ===== Decrypt =====
    # read
    camouflages = [cv2.imread(f"camouflage_{idx}.png") for idx in range(len(camouflages))]
    camouflages_ori = [cv2.imread(f"camouflage_ori_{idx}.png") for idx in range(len(camouflages_ori))]
    rs = vc.img2r(cv2.imread("rs.png", cv2.IMREAD_GRAYSCALE))
    rs_ori = vc.img2r(cv2.imread("rs_ori.png", cv2.IMREAD_GRAYSCALE))

    secret_rec = vc.decrypt_n9(camouflages, rs)
    secret_ori_rec = vc.decrypt_n9(camouflages_ori, rs_ori)
    # save
    cv2.imwrite("secret_recovered.png", secret_rec)
    cv2.imwrite("secret_recovered_ori.png", secret_ori_rec)

    # ===== Evaluation =====
    evaluator = Evaluator()

    # 1. PSNR between secret_rec and secret & secret_ori_rec and secret
    psnr = evaluator.PSNR(secret, secret_rec)
    psnr_ori = evaluator.PSNR(secret, secret_ori_rec)
    print(f"PSNR between secret_rec and secret: {psnr}")
    print(f"PSNR between secret_ori_rec and secret: {psnr_ori}")

    # 2. ISNR between covers and camouflages
    isnr_exp = evaluator.ISNR(covers, camouflages_ori, camouflages, 3, method="expand")
    isnr_mean = evaluator.ISNR(covers, camouflages_ori, camouflages, 3, method="mean")
    isnr_median = evaluator.ISNR(covers, camouflages_ori, camouflages, 3, method="median")
    print(f"ISNR between covers and camouflages (expand): {isnr_exp}")
    print(f"ISNR between covers and camouflages (mean): {isnr_mean}")
    print(f"ISNR between covers and camouflages (median): {isnr_median}")

