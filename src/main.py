import numpy as np
import cv2
import random
from tqdm import tqdm

STDIMAGES = "../standard_images/"
SEED = 2024

# set seed
random.seed(SEED)
np.random.seed(SEED)

class VisualCipher:
    def __init__(self):
        self.lena = cv2.imread(STDIMAGES + "color/lena_color_512.tif", cv2.IMREAD_COLOR)
        self.mandril = cv2.imread(STDIMAGES + "color/mandril_color.tif", cv2.IMREAD_COLOR)
        self.peppers = cv2.imread(STDIMAGES + "color/peppers_color.tif", cv2.IMREAD_COLOR)

    def construct_S_29(self, k, r):
        m, n = 9,2
        # r must between 1 and m
        assert 1 <= r and r <= m, "r must satisfy 1 <= r <= m"

        S = np.zeros((n, m), dtype=np.uint8)
        # convert k into bits
        k_bits = np.unpackbits(np.array([k], dtype=np.uint8))

        # randomly assign first row, with r-1-th element being 1
        # rest of the columns have half are 1 and half are 0
        assigns = [0] * (m//2) + [1] * (m//2)
        np.random.shuffle(assigns)

        for j in range(m):
            if j == r-1:
                S[0, j] = 1
                S[1, j] = 0
                continue
            elif j < r-1:
                S[0, j] = assigns[j]
                S[1, j] = S[0, j] ^ k_bits[j]
            elif j > r-1:
                S[0, j] = assigns[j-1]
                S[1, j] = S[0, j] ^ k_bits[j-1]
        return S
    
    def reconstruct_S_29(self, r, patch0, patch1):
        m, n = 9,2
        
        S = np.zeros((n, m), dtype=np.uint8)

        # replace the larger value in S with 1 and the smaller value with 0 in each row
        S0 = patch0.copy().flatten()
        one_value = S0[r-1]
        S0[S0 == one_value] = 1
        S0[S0 != 1] = 0

        S1 = patch1.copy().flatten()
        zero_value = S1[r-1]
        S1[S1 == zero_value] = 0
        S1[S1 != 0] = 1

        S[0] = S0
        S[1] = S1

        return S

    def recover_pixel_29(self, S, r):
        m, n = 9, 2
        # r must between 1 and m
        assert 1 <= r and r <= m, "r must satisfy 1 <= r <= m"

        k = 0
        for j in range(m):
            if j == r-1:
                continue
            elif j < r-1:
                k += (S[0,j] ^ S[1,j]) * (2 ** (7-j))
            elif j > r-1:
                k += (S[0,j] ^ S[1,j]) * (2 ** (7-j+1))
        return k
    
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

    def encrypt_29(self, secret, covers):
        m, n = 9, 2
        h, w, c = secret.shape
        scale = int(m ** 0.5)

        # length of covers must be n
        assert len(covers) == n, "length of covers must be 2"
        # all cover images must share same shape
        for cover in covers:
            assert cover.shape == secret.shape, "all cover images must share same shape"

        # add 1 to all cover images
        for cover in covers:
            cover += 1
            # if exceed 255, set to 255
            cover[cover > 255] = 255

        # generate random number in shape of (h, w)
        rs = np.random.randint(1, m+1, (h, w))

        # camouflages
        camouflages = []
        for _ in range(n):
            camouflages.append(np.zeros((h * scale, w * scale, c), dtype=np.uint8))

        pbar = tqdm(total=h*w*c, desc="Encrypting")

        for z in range(c):
            for i in range(h):
                for j in range(w):
                    r = rs[i, j]
                    k = secret[i, j, z]
                    S = self.construct_S_29(k, r)

                    # if (i,j) in [(10, 10), (11, 20), (100, 209)]:
                    #     print(i, j)
                    #     print(r)
                    #     print(k)
                    #     print(S)
                    
                    S0 = S[0].reshape(scale, scale)
                    S1 = S[1].reshape(scale, scale)

                    # replace 1 in S0 with covers[0][i, j, z] and replace 0 in S0 with covers[1][i, j, z] - 1
                    S0[S0 == 1] = covers[0][i, j, z]
                    S0[S0 == 0] = covers[0][i, j, z] - 1
                    # replace 1 in S1 with covers[1][i, j, z] and replace 0 in S1 with covers[0][i, j, z] - 1
                    S1[S1 == 1] = covers[1][i, j, z]
                    S1[S1 == 0] = covers[1][i, j, z] - 1

                    # put S0 and S1 into camouflages
                    camouflages[0][i*scale:(i+1)*scale, j*scale:(j+1)*scale, z] = S0
                    camouflages[1][i*scale:(i+1)*scale, j*scale:(j+1)*scale, z] = S1

                    pbar.update(1)

        pbar.close()

        return camouflages, rs
    
    def decrypt_29(self, camouflages, rs):
        m, n = 9, 2
        scale = int(m ** 0.5)
        h, w, c = camouflages[0].shape

        secret = np.zeros((h//scale, w//scale, c), dtype=np.uint8)

        pbar = tqdm(total=(h//scale)*(w//scale)*c, desc="Decrypting")

        for z in range(c):
            for i in range(h//scale):
                for j in range(w//scale):
                    r = rs[i, j]
                    S = self.reconstruct_S_29(r, camouflages[0][i*scale:(i+1)*scale, j*scale:(j+1)*scale, z], camouflages[1][i*scale:(i+1)*scale, j*scale:(j+1)*scale, z])
                    k = self.recover_pixel_29(S, r)

                    # if (i,j) in [(10, 10), (11, 20), (100, 209)]:
                    #     print(i, j)
                    #     print(r)
                    #     print(k)
                    #     print(S)

                    secret[i, j, z] = k
                    pbar.update(1)

        pbar.close()

        return secret

if __name__ == "__main__":
    vc = VisualCipher()
    secret = vc.lena
    covers = [vc.mandril, vc.peppers]

    camouflages, rs = vc.encrypt_29(secret, covers)

    # save
    cv2.imwrite("camouflage0.png", camouflages[0])
    cv2.imwrite("camouflage1.png", camouflages[1])        
    cv2.imwrite("rs.png", vc.r2img(rs))  

    rs = vc.img2r(cv2.imread("rs.png", cv2.IMREAD_GRAYSCALE))
    secret_recovered = vc.decrypt_29(camouflages, rs)

    # save
    cv2.imwrite("secret_recovered.png", secret_recovered)