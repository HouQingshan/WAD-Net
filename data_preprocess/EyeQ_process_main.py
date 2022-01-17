import data_preprocess.fundus_prep as prep
import glob
import os
import cv2 as cv
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process(image_list, save_path):
	for image_path in image_list:
		dst_image = os.path.splitext(image_path.split('\\')[-1])[0] + '.png'
		print(dst_image)
		dst_path = os.path.join(save_path, dst_image)
		print(save_path)
		print(dst_path)
		if os.path.exists(dst_path):
			print('continue...')
			continue
		try:
			img = prep.imread(image_path)
			r_img, borders, mask = prep.process_without_gb(img)
			r_img = cv.resize(r_img, (2000, 2000))
			prep.imwrite(dst_path, r_img)
		except:
			print(image_path)
			continue


if __name__ == "__main__":
	
	image_list = glob.glob(os.path.join('C:/Users/15383/Desktop/Pro/data_preprocess/Original_Img', '*.jpeg'))
	save_path = prep.fold_dir('C:/Users/15383/Desktop/Pro/data_preprocess/Progress_Img')
	
	process(image_list, save_path)
