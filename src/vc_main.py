import argparse
from vc_9 import VisualCipher as VisualCipher9
from vc_16 import VisualCipher as VisualCipher16
import cv2
import numpy as np
import random
import os

def read_image(path, color=True):
    if color:
        return cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def write_image(image, path):
    cv2.imwrite(path, image)

def white_image(h, w, c):
    return 255 * np.ones((h, w, c), dtype=np.uint8)

def random_image(h, w, c):
    if c == 1:
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)
    else:
        return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="Encrypt and decrypt of visual cryptography scheme", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--encrypt", action="store_true", help="Encrypt mode")
    parser.add_argument("--decrypt", action="store_true", help="Decrypt mode")
    parser.add_argument("--gray", action="store_true", help="Use grayscale secret images")

    parser.add_argument("--secrets", type=str, nargs="+", help="Paths of secret images")
    parser.add_argument("--nsecrets", type=int, help="Number of secret images, for decryption")
    parser.add_argument("--covers", type=str, nargs="+", help="Paths of cover images for encryption, or paths of camouflage images for decryption")
    parser.add_argument("--output", type=str, default="./output/", help="Output directory")
    parser.add_argument("--r", type=str, help="Path of the random number r image")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation")

    args = parser.parse_args()

    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # check if the arguments are valid
    if args.encrypt and args.decrypt:
        parser.error("Cannot encrypt and decrypt at the same time")
    if not args.encrypt and not args.decrypt:
        parser.error("Must specify either encrypt or decrypt mode")
    if args.encrypt and not args.secrets:
        parser.error("Must provide secret images for encryption")

    # covers must be at least 2 when encrypting
    if args.encrypt and len(args.covers) < 2:
        parser.error("Must provide at least 2 cover images for encryption")

    if args.decrypt and len(args.covers) < 2:
        parser.error("Must provide at least 2 camouflage images for decryption")

    # if is decrypting, must provide r
    if args.decrypt and not args.r:
        parser.error("Must provide r for decryption")

    # if is decrypting, must provide nsecrets
    if args.decrypt and not args.nsecrets:
        parser.error("Must provide nsecrets for decryption")

    # if output path does not end with "/", add it
    if args.output[-1] != "/":
        args.output += "/"

    # if is color secret images, the number of secret images must be 1 or 2
    if args.encrypt and not args.gray and len(args.secrets) > 2:
        parser.error("Cannot have more than 2 color secret images")

    # if is grayscale secret images, the number of secret images must less than 6
    if args.encrypt and args.gray and len(args.secrets) > 6:
        parser.error("Cannot have more than 6 grayscale secret images")

    # if the number of grayscale images is not the multiple of 3, warn users taht we'll add white images to make it a multiple of 3
    # if args.encrypt and args.gray and len(args.secrets) % 3 != 0:
    #     print("Warning: The number of grayscale images is not the multiple of 3. Random noise images will be added to make it a multiple of 3 for encryption.\nWhen decrypting, remember to set the number of secret images same as current number of secret images, otherwise, there will be extra noise images in the output.")

    # if is decrypting, the number of secrets must be <= 2 if color, or <= 6 if grayscale
    if args.decrypt and not args.gray and args.nsecrets > 2:
        parser.error("Cannot have more than 2 color secret images for decryption")
    if args.decrypt and args.gray and args.nsecrets > 6:
        parser.error("Cannot have more than 6 grayscale secret images for decryption")

    return args

def create_output_dir(output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    args = parse_args()

    create_output_dir(args.output)
    mode = "encrypt" if args.encrypt else "decrypt"
    is_gray = args.gray

    covers = [read_image(cover) for cover in args.covers]

    # check if all covers have the same shape
    h, w, c = covers[0].shape
    for cover in covers:
        assert cover.shape[0] == h and cover.shape[1] == w, "Error: All covers must have the same shape"

    if mode == "encrypt":
        secrets = args.secrets
        n_secrets = len(secrets)
        secrets = [read_image(secret, color=(not is_gray)) for secret in args.secrets]

        # check if all secret images have the same shape
        for secret in secrets:
            assert secret.shape[0] == h and secret.shape[1] == w, "Error: All secret and cover images must have the same shape"

    if mode == "encrypt" and is_gray and n_secrets % 3 != 0:
        n_white = 3 - n_secrets % 3
        for _ in range(n_white):
            secrets.append(random_image(secrets[0].shape[0], secrets[0].shape[1], 1))

    if mode == "encrypt" and is_gray:
        real_secrets = []
        # put 3 grayscale images together to form a color image
        for i in range(len(secrets) // 3):
            secret = np.stack(secrets[i*3:i*3+3], axis=-1)
            real_secrets.append(secret)
    elif mode == "encrypt":
        real_secrets = secrets

    n_secrets = len(real_secrets) if mode == "encrypt" else args.nsecrets


    if mode == "encrypt":
        if n_secrets == 1:
            vc = VisualCipher9()
            camouflages, rs = vc.encrypt_n9(real_secrets[0], covers)
            for i, camouflage in enumerate(camouflages):
                write_image(camouflage, f"{args.output}camouflage_{i}.png")
            write_image(vc.r2img(rs), f"{args.output}rs.png")
        elif n_secrets == 2:
            vc = VisualCipher16()
            camouflages, rs = vc.encrypt_n16(real_secrets[0], real_secrets[1], covers)
            for i, camouflage in enumerate(camouflages):
                write_image(camouflage, f"{args.output}camouflage_{i}.png")
            write_image(vc.r2img(rs), f"{args.output}rs.png")

    elif mode == "decrypt":
        if not is_gray:
            if n_secrets == 1:
                vc = VisualCipher9()
                rs = vc.img2r(read_image(args.r, color=False))
                secret = vc.decrypt_n9(covers, rs)
                write_image(secret, f"{args.output}secret_0_recovered.png")
            elif n_secrets == 2:
                vc = VisualCipher16()
                rs = vc.img2r(read_image(args.r, color=False))
                secret1, secret2 = vc.decrypt_n16(covers, rs)
                write_image(secret1, f"{args.output}secret_0_recovered.png")
                write_image(secret2, f"{args.output}secret_1_recovered.png")

        else:
            if n_secrets <= 3:
                vc = VisualCipher9()
                rs = vc.img2r(read_image(args.r, color=False))
                secret = vc.decrypt_n9(covers, rs)
                for i in range(n_secrets):
                    secret_gray = secret[:, :, i]
                    write_image(secret_gray, f"{args.output}secret_{i}_recovered.png")

            elif n_secrets <= 6:
                vc = VisualCipher16()
                rs = vc.img2r(read_image(args.r, color=False))
                secret1, secret2 = vc.decrypt_n16(covers, rs)
                secrets_gray = [secret1[:, :, i] for i in range(3)]
                secrets_gray += [secret2[:, :, i] for i in range(n_secrets - 3)]

                for i, secret_gray in enumerate(secrets_gray):
                    write_image(secret_gray, f"{args.output}secret_{i}_recovered.png")


if __name__ == "__main__":
    main()
