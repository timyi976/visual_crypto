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

## Usage Examples

- Hide 2 color secret images into 4 cover images

	```bash
	# run under src/
	# make sure src/output/ exists
	
	# encryption
	python3 vc_main.py --encrypt --secrets ../standard_images/color/lenna_color_512.tif ../standard_images/color/barbara.tif --covers ../standard_images/color/lake_color.tif ../standard_images/color/lighthouse.tif ../standard_images/color/peppers_color.tif ../standard_images/color/mandril_color.tif --seed 4096

	# decryption
	python3 vc_main.py --decrypt --covers ./output/camouflage_0.png ./output/camouflage_1.png ./output/camouflage_2.png ./output/camouflage_3.png --nsecrets 2 --r ./output/rs.png
	```

- Hide 3 grayscale secret images into 3 cover images

	```bash
	# run under src/
	# make sure src/output/ exists
	
	# encryption
	python3 vc_main.py --encrypt --gray --secrets ../standard_images/gray/cameraman.tif ../standard_images/gray/house.tif ../standard_images/gray/jetplane.tif --covers ../standard_images/color/barbara.tif ../standard_images/color/lake_color.tif ../standard_images/color/lenna_color_512.tif

	# decryption
	python3 vc_main.py --decrypt --gray --covers ./output/camouflage_0.png ./output/camouflage_1.png ./output/camouflage_2.png --nsecrets 3 --r ./output/rs.png
	```

## Reference

- [1] Chang, C. C. and Yu. T. X., Sharing a Secret Gray Image in Multiple Images, in the Proceedings of International Symposium on Cyber Worlds: Theories and Practice, Tokyo, Japan, Nov. 2002, pp.230-237

- [2] Youmaran, R., Adler, A., and Miri, A. An Improved Visual Cryptography Scheme for Secret Hiding. In23rd Biennial Symposium on Communications,2006(May 2006), pp. 340–343