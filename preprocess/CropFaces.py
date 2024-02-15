import os
from CropFaceAndAlign import CropFace
from PIL import Image

db_path = "C:\\Users\\amitc\\Downloads\\NIR-VIS-2.0"
subfolders = ["s1", "s2", "s3", "s4"]
nir_vis_eyes_files = ["vis_eyes.txt", "nir_eyes.txt"]
for subfolder in subfolders:
	subfolder_full_path = db_path + "\\" + subfolder
	# employees = []
	# for r, d, f in os.walk(db_path + "/" + subfolder): # r=root, d=directories, f = files
	#	for file in f:
	#		if (('.jpg' in file or '.bmp' in file) and "128x128" not in r):
	#			exact_path = r + "/" + file
	#			employees.append(exact_path)

	# print("Aligning and cropping " + len(employees) + " images")
	for eyes_file in nir_vis_eyes_files:
		eyes_file_full_name = subfolder + "_" + eyes_file
		with open(subfolder_full_path + "\\" + eyes_file_full_name, 'r') as read_obj:
			for line in read_obj:
				#split_line = line.strip().replace("\\","/").split(" ")
				split_line = line.strip().split(" ")
				image_path = subfolder_full_path + "\\" + split_line[0]
				aligned_image_path = subfolder_full_path + "\\" + "128_128_" + split_line[0]

				directory = os.path.dirname(aligned_image_path)
				if not os.path.exists(directory):
					os.makedirs(directory)

				image = Image.open(image_path)
				CropFace(image, eye_left=(int(split_line[1]),int(split_line[2])), eye_right=(int(split_line[3]),int(split_line[4])), dest_sz=(224,224)).save(aligned_image_path)
