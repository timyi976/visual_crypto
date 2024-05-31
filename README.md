# Visual Cryptography

Visual Cryptography, Final Project for Digital Image Processing Course of NTU CSIE

## Usage

```
usage: python3 vc_main.py [-h] [--encrypt] [--decrypt] [--gray] [--secrets SECRETS [SECRETS ...]] [--nsecrets NSECRETS] [--covers COVERS [COVERS ...]] [--output OUTPUT] [--r R]
                  [--seed SEED]

Encrypt and decrypt of visual cryptography scheme

options:
  -h, --help            show this help message and exit
  --encrypt             Encrypt mode
  --decrypt             Decrypt mode
  --gray                Use grayscale secret images
  --secrets SECRETS [SECRETS ...]
                        Paths of secret images
  --nsecrets NSECRETS   Number of secret images, for decryption
  --covers COVERS [COVERS ...]
                        Paths of cover images for encryption, or paths of camouflage images for decryption
  --output OUTPUT       Output directory
  --r R                 Path of the random number r image
  --seed SEED           Seed for random number generation
```

## Reference

- [1] Chang, C. C. and Yu. T. X., Sharing a Secret Gray Image in Multiple Images, in the Proceedings of International Symposium on Cyber Worlds: Theories and Practice, Tokyo, Japan, Nov. 2002, pp.230-237

- [2] Youmaran, R., Adler, A., and Miri, A. An Improved Visual Cryptography Scheme for Secret Hiding. In23rd Biennial Symposium on Communications,2006(May 2006), pp. 340–343