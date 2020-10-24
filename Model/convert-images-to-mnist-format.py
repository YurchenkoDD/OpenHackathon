
###
# Скрипт для конвертирования изображений датасета в формат mnist
# Доработка скрипта : https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format
# под новые версии библиотек и под работу с конкретным датасетом
###

import os
from PIL import Image
from array import *
import json
import argparse as ap


def getOutImagesName(prefix, total):
	return prefix+'-images-idx3-ubyte-'+str(total)


def getOutLabelsName(prefix, total):
	return prefix+'-labels-idx1-ubyte-'+str(total)


def run(args):
	trainPrefix = 'train'
	Names = [[args['trainFolder'], trainPrefix]]
	if args['testFolder'] is not None:
		testPrefix = args.testPrefix if args.testPrefix is not None else 'test'
		Names.append([args['testFolder'], args['textPrefix']])
		
	total = 0
	for name in Names:
		
		data_image = array('B')
		data_label = array('B')

		FileList = []
		for dirname in os.listdir(name[0])[0:]:

			path = os.path.join(name[0],dirname)
		
			for filename in os.listdir(path):
				if filename.endswith(".png") :
					FileList.append(os.path.join(path, filename))

		errors = []

		for filename in FileList:
			try:
				label = int(os.path.basename(os.path.abspath(os.path.join(filename,os.path.pardir))))
				f = os.path.abspath(filename)
				print(f)
				Im = Image.open(f)
				Im = Im.convert('1')
				pixel = Im.load()

				width, height = Im.size

				for x in range(0,width):
					for y in range(0,height):
						data_image.append(pixel[y,x])

				data_label.append(label)
				total +=1
			except Exception as e:
				errors.append(json.dumps({"err": str(e), "fn": filename}))

		if len(errors) > 0:
			print( len(errors) , "occurred")
			with open("errors.json",'w') as errfile:
				errfile.write(json.dumps(errors))	

		hexval = "{0:#0{1}x}".format(len(FileList),10)
		header = array('B')
		header.extend([0,0,8,1])
		header.append(int('0x'+hexval[2:][:2],16))
		header.append(int('0x'+hexval[4:][:2],16))
		header.append(int('0x'+hexval[6:][:2],16))
		header.append(int('0x'+hexval[8:][:2],16))

		data_label = header + data_label
		width = 28
		height = 28
		hexval = "{0:#0{1}x}".format(width,10)
		header.append(int('0x'+hexval[2:][:2],16))
		header.append(int('0x'+hexval[4:][:2],16))
		header.append(int('0x'+hexval[6:][:2],16))
		header.append(int('0x'+hexval[8:][:2],16))
		hexval = "{0:#0{1}x}".format(height,10)
		header.append(int('0x'+hexval[2:][:2],16))
		header.append(int('0x'+hexval[4:][:2],16))
		header.append(int('0x'+hexval[6:][:2],16))
		header.append(int('0x'+hexval[8:][:2],16))

		header[3] = 3
		data_image = header + data_image

		output_file = open(getOutImagesName(trainPrefix, total) , 'wb')
		data_image.tofile(output_file)
		output_file.close()

		output_file = open(getOutLabelsName(trainPrefix, total), 'wb')
		data_label.tofile(output_file)
		output_file.close()

		os.system('gzip '+ getOutImagesName(trainPrefix, total))
		os.system('gzip '+ getOutLabelsName(trainPrefix, total))


if __name__ == "__main__":
	parser = ap.ArgumentParser()
	parser.add_argument("-p", "--trainFolder", help="Path to training images", required=True)
	parser.add_argument("-q", "--trainPrefix", help="Prefix of name of training output")
	parser.add_argument("-r", "--testFolder", help="Path to test images")
	parser.add_argument("-s", "--testPrefix", help="Prefix of name of test output")
	parser.add_argument("-b", "--bw", help="Convert images to black and white", action="store_true")
	args = vars(parser.parse_args())
	print("Args:", args)
	run(args)
