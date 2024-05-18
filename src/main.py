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

class VisualCipher:
    def __init__(self):
        self.lena = cv2.imread(STDIMAGES + "color/lena_color_512.tif", cv2.IMREAD_COLOR)
        self.mandril = cv2.imread(STDIMAGES + "color/mandril_color.tif", cv2.IMREAD_COLOR)
        self.peppers = cv2.imread(STDIMAGES + "color/peppers_color.tif", cv2.IMREAD_COLOR)

        # for m * n scheme
        self.airplaneF16 = cv2.imread(STDIMAGES + "color/airplaneF16.tif", cv2.IMREAD_COLOR)
        self.cameraman = cv2.imread(STDIMAGES + "gray/cameraman.tif", cv2.IMREAD_COLOR)
        self.house = cv2.imread(STDIMAGES + "gray/house.tif", cv2.IMREAD_COLOR)
        self.jetplane = cv2.imread(STDIMAGES + "gray/jetplane.tif", cv2.IMREAD_COLOR)
        self.lake = cv2.imread(STDIMAGES + "gray/lake.tif", cv2.IMREAD_COLOR)

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

    def construct_S_2_out_of_2(self, k, r, need_unpack_bits=True):
        m, n = 9, 2
        assert 1 <= r <= m, "r must satisfy 1 <= r <= m"

        #S = np.zeros((n, m), dtype=np.uint8)
        # Use 3 to represent null elements
        S = np.full((n, m), 3, dtype=np.uint8)
        
        noOfOnes = 0

        # convert k into bits
        if need_unpack_bits:
            k_bits = np.unpackbits(np.array([k], dtype=np.uint8))
        else:
            k_bits = k

        for i in range(m - 1):
            if i < r-1 and k_bits[i] == 1:
                noOfOnes += 1
                if noOfOnes % 2 == 0:
                    S[0, i] = 0
                else:
                    S[0, i] = 1
            elif i >= r-1 and k_bits[i] == 1:
                noOfOnes += 1
                j = i + 1
                if noOfOnes % 2 == 1:
                    S[0, j] = 1
                else:
                    S[0, j] = 0

        if noOfOnes % 2 == 1:
            S[0, r-1] = 0

        # Randomly assign the rest null elements in row 1
        # (To simplify the problem, we assign 1 to the null in row 1)
        for i in range(m):
            if S[0,i] == 3:
                S[0,i] = 1

        #Compute row 2

        j = -1
        for i in range(m):
            if i != r-1:
                j += 1
                S[1,i] = S[0,i] ^ k_bits[j]
            else:
                S[1,i] = S[0,i] ^ (noOfOnes % 2)
                
        return S

    def construct_S_mn(self, k, r, m, n):
        # r must be between 1 and m
        assert 1 <= r <= m, "r must satisfy 1 <= r <= m"
        
        S = np.zeros((n, m), dtype=np.uint8)

        # Convert k into bits
        k_bits = np.unpackbits(np.array([k], dtype=np.uint8))

        # Random assign n-2 rows with five 1s and four 0s
        
        # Randomly choose 2 non_assign rows
        choose_none_assign_row = np.random.choice(n, 2, replace=False)
        
        for i in range(n):
            if i in choose_none_assign_row:
                continue
            # Randomly choose m//2+1 indices to place 1s
            ones_indices = np.random.choice(m, m // 2 + 1, replace=False)
            S[i, ones_indices] = 1

        # Compute t
        t = np.zeros(m, dtype=np.uint8)

        for i in range(m):
            if i < r - 1:
                t[i] = k_bits[i]
            elif i == r - 1:
                t[i] = np.random.randint(0, 2)
            else:
                t[i] = k_bits[i - 1]

        # Compute t0
        t0 = np.bitwise_xor(t, np.sum(S, axis=0) % 2)

        # Apply the uniform 2 out of 2 scheme to compute row S_none_assign1 and S_none_assign2
        k0 = np.delete(t0, r - 1)
        two_out_of_two = self.construct_S_2_out_of_2(k0, r, need_unpack_bits=False)


        # check t equal S_1 xor S_2 xor S_3
        S[choose_none_assign_row[0]] = two_out_of_two[0]
        S[choose_none_assign_row[1]] = two_out_of_two[1]

        return S
    
    def encrypt_mn(self, secret, covers, m, n, to_improve=True):
        h, w, c = secret.shape
        scale = int(m ** 0.5)

        # Length of covers must be n
        assert len(covers) == n, "Length of covers must be n"
        # All cover images must share the same shape
        for cover in covers:
            assert cover.shape == secret.shape, "All cover images must share the same shape"

        # Add 1 to all cover images
        for cover in covers:
            cover += 1
            # If exceeding 255, set to 255
            cover[cover > 255] = 255

        # Make deep copies of covers to ensure original covers are not modified
        covers_copy = [copy.deepcopy(cover) for cover in covers]

        if to_improve:
            # Add 1 to all cover images
            for cover in covers_copy:
                cover += 1
                # If exceeding 255, set to 255
                cover[cover > 255] = 255
        else:
            # Ensure distinct value separation for non-improved version
            for cover in covers_copy:
                cover[cover < 128] += 1
                cover[cover >= 128] -= 1

        # Generate random numbers in shape of (h, w)
        rs = np.random.randint(1, m + 1, (h, w))

        # Camouflages
        camouflages = []
        for _ in range(n):
            camouflages.append(np.zeros((h * scale, w * scale, c), dtype=np.uint8))

        pbar = tqdm(total=h * w * c, desc="Encrypting")

        for z in range(c):
            for i in range(h):
                for j in range(w):
                    r = rs[i, j]
                    k = secret[i, j, z]
                    S = self.construct_S_mn(k, r, m, n)
                    for share_index in range(n):
                        S_share = S[share_index].reshape(scale, scale)
                        # Replace 1 in S_share with covers_copy[share_index][i, j, z] 
                        # Replace 0 in S_share with covers_copy[share_index][i, j, z] - 1
                        S_share[S_share == 1] = covers_copy[share_index][i, j, z]
                        if to_improve:
                            S_share[S_share == 0] = covers_copy[share_index][i, j, z] - 1
                        else:
                            S_share[S_share == 0] = 0

                        # Put S_share into camouflages
                        camouflages[share_index][i * scale:(i + 1) * scale, j * scale:(j + 1) * scale, z] = S_share

                        pbar.update(1)

        pbar.close()

        return camouflages, rs

    def reconstruct_S_mn(self, r, Camouflages, scale, i, j, z, m, n):
        
        S = np.zeros((n, m), dtype=np.uint8)

        
        # replace the larger value in S with 1 and the smaller value with 0 in each row
        for share_index in range(n):
            patch = Camouflages[share_index][i * scale:(i + 1) * scale, j * scale:(j + 1) * scale, z]
            S_share = patch.copy().flatten()
            one_value = max(S_share)
            S_share[S_share == one_value] = 1
            S_share[S_share != 1] = 0
            S[share_index] = S_share

        #print(S)

        return S

    def recover_pixel_mn(self, S, r, m, n):
        # r must between 1 and m
        assert 1 <= r and r <= m, "r must satisfy 1 <= r <= m"

        T = np.sum(S, axis=0) % 2


        k = 0
        for j in range(m):
            if j == r-1:
                continue
            elif j < r-1:
                # t_i = k_i
                k += T[j] * (2 ** (7-j))
                #k += (S[0,j] ^ S[1,j]) * (2 ** (7-j))
            elif j > r-1:
                # t_i = k_i-1
                k += T[j] * (2 ** (7-j+1))
                #k += (S[0,j] ^ S[1,j]) * (2 ** (7-j+1))
        return k
    
    def decrypt_mn(self, camouflages, rs, m, n, to_improve=True):
        scale = int(m ** 0.5)
        h, w, c = camouflages[0].shape
        
        # Initialize container for the decrypted secret image
        secret = np.zeros((h // scale, w // scale, c), dtype=np.uint8)

        pbar = tqdm(total=(h // scale) * (w // scale) * c, desc="Decrypting")

        for z in range(c):
            for i in range(h // scale):
                for j in range(w // scale):
                    r = rs[i, j]
                    #S = np.zeros((n, m), dtype=np.uint8)
                    S = self.reconstruct_S_mn(r, camouflages, scale, i, j, z, m, n)

                    
                    # Recover original pixel value
                    k = self.recover_pixel_mn(S, r, m, n)
                    secret[i, j, z] = k
                    pbar.update(1)

        pbar.close()

        return secret
    
    # PSNR
    def PSNR(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100

        return 10 * np.log10(255**2 / mse)
    
if __name__ == "__main__":
    vc = VisualCipher()
    secret = vc.lena
    covers = [vc.mandril, vc.peppers, vc.airplaneF16]
    '''
    camouflages, rs = vc.encrypt_29(secret, covers)

    # save
    cv2.imwrite("camouflage0.png", camouflages[0])
    cv2.imwrite("camouflage1.png", camouflages[1])        
    cv2.imwrite("rs.png", vc.r2img(rs))  

    rs = vc.img2r(cv2.imread("rs.png", cv2.IMREAD_GRAYSCALE))
    secret_recovered = vc.decrypt_29(camouflages, rs)

    # save
    cv2.imwrite("secret_recovered.png", secret_recovered)

    secret = vc.cameraman
    covers = [vc.house, vc.jetplane, vc.lake]
    '''
    m = 9
    n = 3
    camouflages_old, rs_old = vc.encrypt_mn(secret, covers, m, n, to_improve=False)
    camouflages_improved, rs_improved = vc.encrypt_mn(secret, covers, m, n, to_improve=True)
    # save
    for i in range(n):
        cv2.imwrite(f"camouflage{i}_old.png", camouflages_old[i])
        cv2.imwrite(f"camouflage{i}_improved.png", camouflages_improved[i])

    cv2.imwrite("rs_old.png", vc.r2img(rs_old))
    cv2.imwrite("rs_improved.png", vc.r2img(rs_improved))
    
    rs_old = vc.img2r(cv2.imread("rs_old.png", cv2.IMREAD_GRAYSCALE))
    rs_improved = vc.img2r(cv2.imread("rs_improved.png", cv2.IMREAD_GRAYSCALE))

    secret_recovered_old = vc.decrypt_mn(camouflages_old, rs_old, m, n, to_improve=False)
    secret_recovered_improved = vc.decrypt_mn(camouflages_improved, rs_improved, m, n)
    # save
    cv2.imwrite("secret_recovered_mn.png", secret_recovered_old)
    cv2.imwrite("secret_recovered_mn_improved.png", secret_recovered_improved)
    print(f'PSNR secret_recovered_mn:{vc.PSNR(secret, secret_recovered_old)}')
    print(f'PSNR secret_recovered_mn_improved:{vc.PSNR(secret, secret_recovered_improved)}')
