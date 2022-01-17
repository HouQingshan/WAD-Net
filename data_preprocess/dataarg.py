import cv2
import numpy as np
import os


# 图像平移
def img_translation(image, item, step=4):
	# 图像平移 下、上、右、左平移
	M = np.float32([[1, 0, 0], [0, 1, 100]])
	img_down = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
	
	M = np.float32([[1, 0, 0], [0, 1, -100]])
	img_up = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
	
	M = np.float32([[1, 0, 100], [0, 1, 0]])
	img_right = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
	
	M = np.float32([[1, 0, -100], [0, 1, 0]])
	img_left = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
	
	#
	for i in range(step):
		if i == 0:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, img_down)
		elif i == 1:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, img_up)
		elif i == 2:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, img_right)
		else:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, img_left)


# #
# cv2.imshow("down", img_down)
# cv2.imshow("up", img_up)
# cv2.imshow("right", img_right)
# cv2.imshow("left", img_left)


#
def img_scale(image):
	result = cv2.resize(image, (224, 224))
	cv2.imshow("scale", result)


# filename = ' xxx ' + '.jpeg'
# cv2.imwrite(savedpath + filename, result)


#
def img_flip(image, item, step=3):
	#
	horizontally = cv2.flip(image, 0)  #
	vertically = cv2.flip(image, 1)  #
	hv = cv2.flip(image, -1)  #
	for i in range(4, 4 + step):
		if i == 4:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, horizontally)
		elif i == 5:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, vertically)
		
		else:
			filename = item[:-4] + '_arg' + str(i) + '.jpg'
			cv2.imwrite(savedpath + filename, hv)


#
# cv2.imshow("Horizontally", horizontally)
# cv2.imshow("Vertically", vertically)
# cv2.imshow("Horizontally & Vertically", hv)


#
def img_rotation(image, item, step=1):
	#
	rows, cols, channel = image.shape
	
	#
	#
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
	#
	rotated = cv2.warpAffine(image, M, (cols, rows))
	
	for i in range(7, 7 + step):
		filename = item[:-4] + '_arg' + str(i) + '.jpg'
		cv2.imwrite(savedpath + filename, rotated)


#
# cv2.imshow("rotated", rotated)


#
def img_noise(image, mean=0, var=0.001, item=None, step=1):
	
	
	image = np.array(image / 255, dtype=float)
	noise = np.random.normal(mean, var ** 0.5, image.shape)
	out = image + noise
	if out.min() < 0:
		low_clip = -1.
	else:
		low_clip = 0.
	out = np.clip(out, low_clip, 1.0)
	out = np.uint8(out * 255)
	for i in range(8, 8 + step):
		filename = item[:-4] + '_arg' + str(i) + '.jpg'
		cv2.imwrite(savedpath + filename, out)

# cv2.imshow("noise", out)


#
def img_brightness(image, item=None, step=1):
	contrast = 1  #
	brightness = 100  #
	pic_turn = cv2.addWeighted(image, contrast, image, 0, brightness)
	# cv2.addWeighted(对象,对比度,对象,对比度)
	'''cv2.addWeighted()实现的是图像透明度的改变与图像的叠加'''
	for i in range(9, 9 + step):
		filename = item[:-4] + '_arg' + str(i) + '.jpg'
		cv2.imwrite(savedpath + filename, pic_turn)

# cv2.imshow('bright', pic_turn)  # 显示图片


if __name__ == '__main__':
	
	i = 0
	path = 'C:/Users/15383/Desktop/Pro/data_preprocess/Original_Img/'
	savedpath = 'C:/Users/15383/Desktop/Pro/data_preprocess/Dataarg_Img/'
	
	filelist = os.listdir(path)
	total_num = len(filelist)
	
	for item in filelist:
		number = i + 1
		i = number
		print(i)
		src = cv2.imread(path + item)
		
		img_translation(src, item)
		img_flip(src, item)
		img_rotation(src, item)
		img_noise(src, item=item)
		img_brightness(src, item)
