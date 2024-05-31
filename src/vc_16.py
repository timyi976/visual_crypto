import numpy as np
import cv2
import copy
import random
from tqdm import tqdm

class VisualCipher:
    def __init__(self):
        pass

    def r2img(self, rs, split=8):
        mapping = {i: 255 // split * i for i in range(1, split+1)}
        rs_img = np.zeros(rs.shape, dtype=np.uint8)
        for i in range(1, split+1):
            rs_img[rs == i] = mapping[i]

        return rs_img
    
    def img2r(self, rs_img, split=8):
        mappping = {255 // split * i: i for i in range(1, split+1)}
        rs = np.zeros(rs_img.shape, dtype=np.uint8)
        for i in range(1, split+1):
            rs[rs_img == 255 // split * i] = i

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
        k1_bits = np.concatenate((k_bits[:r], k_bits[2*r:2*r+(8-r)]))
        k2_bits = np.concatenate((k_bits[r:2*r], k_bits[2*r+(8-r):]))

        k1 = np.packbits(k1_bits)[0]
        k2 = np.packbits(k2_bits)[0]

        return k1, k2
    
    def construct_S_216(self, k, r):
        # r must between 1 and 8
        assert 1 <= r <= 8, "r must satisfy 1 <= r <= 8"

        S = np.full((2, 16), 3, dtype=np.uint8)

        num_ones = 0
        k_bits = k

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

        selected = np.random.choice(n, 2, replace=False)

        for i in range(n):
            if i in selected:
                continue
            ones_idx = np.random.choice(16, 8, replace=False)
            S[i, ones_idx] = 1

        k = self.interfold_k(k1, k2, r, unpack=True)

        t_prime = np.bitwise_xor(k, np.sum(S, axis=0) % 2)

        S_left = self.construct_S_216(t_prime, r)

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
        # change type to uint16 to avoid overflow
        covers = [cover.astype(np.uint16) for cover in covers]

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
                        one_idx = S_share == 1
                        zero_idx = S_share == 0
                        S_share[one_idx] = covers[idx][i, j, z]
                        S_share[zero_idx] = covers[idx][i, j, z] - 1

                        camouflages[idx][i*d:(i+1)*d, j*d:(j+1)*d, z] = S_share.copy()

                        pbar.update(1)

        pbar.close()

        return camouflages, rs

    def reconstruct_S_n16(self, patches):
        n = len(patches)
        S = np.zeros((n, 16), dtype=np.uint8)

        patches = [patch.copy().flatten() for patch in patches]

        for i, patch in enumerate(patches):
            one_value = max(patch)
            patch_new = np.where(patch == one_value, 1, 0).copy()
            S[i] = patch_new

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
    from std_images import StandardImages
    from evaluation import Evaluator

    STDIMAGES = "../standard_images/"
    SEED = 2024

    # set seed
    random.seed(SEED)
    np.random.seed(SEED)

    # ===== Prepare standard images =====
    stdimages = StandardImages()
    secret1 = stdimages.lenna
    secret2 = stdimages.lighthouse
    covers = [stdimages.peppers, stdimages.airplaneF16, stdimages.mandril, stdimages.barbara]

    vc = VisualCipher()

    # ===== Encrypting =====
    camouflages, rs_ori = vc.encrypt_n16(secret1, secret2, covers)
    # save
    for i, camouflage in enumerate(camouflages):
        cv2.imwrite(f"camouflage_{i}.png", camouflage)
    cv2.imwrite("rs1.png", vc.r2img(rs_ori))

    # ===== Decrypting =====
    # read
    camouflages = [cv2.imread(f"camouflage_{i}.png") for i in range(len(covers))]
    rs = vc.img2r(cv2.imread("rs1.png", cv2.IMREAD_GRAYSCALE))
    
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