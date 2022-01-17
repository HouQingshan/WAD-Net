from PIL import Image
import os


# 去除IDRiD黑色边界
# source_path = "C:/Users/15383/Desktop/MAProjec/EyeQ/train/"  # 图片来源路径
# save_path = "C:/Users/15383/Desktop/MAProjec/EyeQ/test_process"
# file_names = os.listdir(source_path)
# for i in range(len(file_names)):
# 	image_path = source_path+file_names[i]
# 	im = Image.open(image_path)
# 	# 图片的宽度和高度
# 	img_size = im.size
# 	print("图片宽度和高度分别是{}".format(img_size))
# 	if im.mode == "P":
# 		im = im.convert('RGB')
# 	if im.mode == "RGBA":
# 		im = im.convert('RGB')
# 	'''
# 	裁剪：传入一个元组作为参数
# 	元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
# 	'''
# 	# 截取图片中一块宽和高都是250的
# 	x = 270
# 	y = 0
# 	w = 3440
# 	h = 2848
# 	region = im.crop((x, y, x + w, y + h))
# 	region.save(save_path+file_names[i])

# 把图片分为36块
source_path = "C:/Users/15383/Desktop/Pro/data_preprocess/Progress_Img/"  # 图片来源路径
save_path = "C:/Users/15383/Desktop/Pro/data_preprocess/Savepatch/"
# source_path = "D:/datas/IDRiD4/labelsetSE/train/"  # 图片来源路径
# save_path = "D:/datas/IDRiD4/labelsetSE/train_patch/"

file_names = os.listdir(source_path)
# file_names_ = file_names[0][:-4]
for i in range(len(file_names)):
# for i in range(1):
	image_path = source_path + file_names[i]
	im = Image.open(image_path)

	img_size = im.size
	w = img_size[0]/6
	h = img_size[1]/6
	x = 0
	y = 0
	num = 0
	for j in range(1, 6+1):

		if j == 1:
			for k in range(6):
				num = int(num)+1
				if num<10:
					num = '0'+str(num)
				region = im.crop((x+k*w, y, x+(k+1)*w, y+h))
				region.save(save_path+file_names[i][:-4]+'_'+str(num)+'.jpg')
		elif j == 2:
			for k in range(6):
				num = int(num)+1
				if num<10:
					num = '0'+str(num)
				region = im.crop((x+k*w, h, x+(k+1) * w, 2*h))
				region.save(save_path + file_names[i][:-4] + '_' + str(num) + '.jpg')
		elif j == 3:
			for k in range(6):
				num = int(num)+1
				if num<10:
					num = '0'+str(num)
				region = im.crop((x+k*w, 2*h, x+(k+1) * w, 3*h))
				region.save(save_path + file_names[i][:-4] + '_' + str(num) + '.jpg')
		elif j == 4:
			for k in range(6):
				num = int(num)+1
				if num<10:
					num = '0'+str(num)
				region = im.crop((x+k*w, 3*h, x+(k+1) * w, 4*h))
				region.save(save_path + file_names[i][:-4] + '_' + str(num) + '.jpg')
		elif j == 5:
			for k in range(6):
				num = int(num)+1
				if num<10:
					num = '0'+str(num)
				region = im.crop((x+k*w, 4*h, x+(k+1) * w, 5*h))
				region.save(save_path + file_names[i][:-4] + '_' + str(num) + '.jpg')
		# elif j == 6:
		# 	for k in range(7):
		# 		num = int(num)+1
		# 		if num<10:
		# 			num = '0'+str(num)
		# 		region = im.crop((x+k*w, 5*h, x+(k+1) * w, 6*h))
		# 		region.save(save_path + file_names[i][:-4] + '_' + str(num) + '.jpg')
		else:
			for k in range(6):
				num = int(num)+1
				if num<10:
					num = '0'+str(num)
				region = im.crop((x + k * w, 5*h, x + (k + 1) * w, 6 * h))
				region.save(save_path + file_names[i][:-4] + '_' + str(num) + '.jpg')

# 制作带有label信息的patches
# source_data_path = "D:/datas/IDRiD/datatest_patch/"
# source_label_path = "D:/datas/IDRiD/labelsetSE/test_patch/"
# save_path = "D:/datas/IDRiD/datatest_patch_withlabel/"
# file_names = os.listdir(source_label_path)
# # image_name = file_names[1][:9]
# # order = file_names[1][12:14]
# # check_file = "D:/datas/IDRiD/datatrain_patch_withlabel/False_IDRiD_03_01.jpg"
# # if Image.open(check_file) != None:
# # 	os.remove(check_file)
#
#
# for i in range(len(file_names)):
# 	image_path = source_label_path + file_names[i]
# 	img = Image.open(image_path)
# 	img_size = img.size
# 	black_num = 0
# 	red_num = 0
# 	for x in range(img_size[0]):
# 		for y in range(img_size[1]):
# 			r, g, b = img.getpixel((x,y))
# 			if 0 <= r <= 10 and 0 <= g <= 10 and 0 <= b <= 10:
# 				black_num += 1
# 			else:
# 				red_num += 1
#
# 	if red_num != 0:
# 		image_name = file_names[i][:9]
# 		order = file_names[i][12:14]
# 		file_name_data = image_name+order+'.jpg'
# 		Lesion_label = "True"
# 		image_data_path = source_data_path + file_name_data
# 		im = Image.open(image_data_path)
# 		file_new_name_data = Lesion_label + '_' + file_name_data
# 		check_file = save_path+"False"+'_' + file_name_data
# 		if os.path.exists(check_file):
# 			os.remove(check_file)
# 			im.save(save_path + file_new_name_data)
# 		else:
# 			im.save(save_path+file_new_name_data)
#
# 	else:
# 		image_name = file_names[i][:9]
# 		order = file_names[i][12:14]
# 		file_name_data = image_name+order+'.jpg'
# 		Lesion_label = "False"
# 		image_data_path = source_data_path + file_name_data
# 		im = Image.open(image_data_path)
# 		file_new_name_data = Lesion_label + '_' + file_name_data
# 		check_file = save_path + "True" + '_' + file_name_data
# 		if os.path.exists(check_file):
# 			pass
# 		else:
# 			im.save(save_path + file_new_name_data)