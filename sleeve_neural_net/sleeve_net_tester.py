import cv2
import numpy as np
# import scipy as sp
import os
import xlwt
from collar_classifier_net import collar_images_maker_for_testing, short_collar_images_maker_for_testing, collar_classifier_neural_net

image_file_types = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

crewneck_path = '/home/nate/Desktop/crewnecks_files/'
roundneck_path = '/home/nate/Desktop/roundnecks_files/'
scoopneck_path = '/home/nate/Desktop/scoopnecks_files/'
squareneck_path = '/home/nate/Desktop/squarenecks_files/'
vneck_path = '/home/nate/Desktop/vnecks_files/'

output_csv_file_name = 'collar_testing.csv'

crewneck_images = [f for f in os.listdir(crewneck_path) if os.path.isfile(os.path.join(crewneck_path, f))]
roundneck_images = [f for f in os.listdir(roundneck_path) if os.path.isfile(os.path.join(roundneck_path, f))]
scoopneck_images = [f for f in os.listdir(scoopneck_path) if os.path.isfile(os.path.join(scoopneck_path, f))]
squareneck_images = [f for f in os.listdir(squareneck_path) if os.path.isfile(os.path.join(squareneck_path, f))]
vneck_images = [f for f in os.listdir(vneck_path) if os.path.isfile(os.path.join(vneck_path, f))]

crewneck_image_files = []
for image_file_name in crewneck_images:
    for image_type in image_file_types:
        if image_type in image_file_name:
            image = cv2.imread(crewneck_path + image_file_name, 1)
            images = collar_images_maker_for_testing(image)
            if len(images) > 0:
                crewneck_image_files.append(images)
            break

roundneck_image_files = []
for image_file_name in roundneck_images:
    for image_type in image_file_types:
        if image_type in image_file_name:
            image = cv2.imread(crewneck_path + image_file_name, 1)
            images = collar_images_maker_for_testing(image)
            if len(images) > 0:
                roundneck_image_files.append(images)
            break

scoopneck_image_files = []
for image_file_name in scoopneck_images:
    for image_type in image_file_types:
        if image_type in image_file_name:
            image = cv2.imread(crewneck_path + image_file_name, 1)
            images = collar_images_maker_for_testing(image)
            if len(images) > 0:
                scoopneck_image_files.append(images)
            break

squareneck_image_files = []
for image_file_name in squareneck_images:
    for image_type in image_file_types:
        if image_type in image_file_name:
            image = cv2.imread(squareneck_path + image_file_name, 1)
            images = collar_images_maker_for_testing(image)
            if len(images) > 0:
                squareneck_image_files.append(images)
            break

vneck_image_files = []
for image_file_name in vneck_images:
    for image_type in image_file_types:
        if image_type in image_file_name:
            image = cv2.imread(vneck_path + image_file_name, 1)
            images = collar_images_maker_for_testing(image)
            if len(images) > 0:
                vneck_image_files.append(images)
            break



crewneck_results = collar_classifier_neural_net(crewneck_image_files)
roundneck_results = collar_classifier_neural_net(roundneck_image_files)
scoopneck_results = collar_classifier_neural_net(scoopneck_image_files)
squareneck_results = collar_classifier_neural_net(squareneck_image_files)
vneck_results = collar_classifier_neural_net(vneck_image_files)

necktypes_result = [crewneck_results,
                   roundneck_results,
                   scoopneck_results,
                   squareneck_results,
                   vneck_results]
necktype_names = ['crewneck results',
                  'roundneck results',
                  ' scoopneckresults',
                  'squareneck results',
                  'vneck results']

columns = list(necktypes_result[0][0])
workbook = xlwt.Workbook()
for k in range(len(necktypes_result)):
    sheet = workbook.add_sheet(necktype_names[k])
    # write headers in row 0
    for j, c in enumerate(columns):
        sheet.write(0, j, c)
    # write columns, start from row 1
    for i, row in enumerate(necktypes_result[k], 1):
        for j, col in enumerate(columns):
            print j
            print col
            sheet.write(i, j, row[col])
workbook.save('collar_classifier_results.xlsx')



# # write headers in row 0
# for j, col in enumerate(columns):
#     ws.write(0, j, col)
#
# # write columns, start from row 1
# for i, row in enumerate(data, 1):
#     for j, col in enumerate(columns):
#         ws.write(i, j, row[col])









# d = {'a':['e1','e2','e3'],'b':['e1','e2'],'c':['e1']}
# row = 0
# col = 0
#
# for key in d.keys():
#     row += 1
#     worksheet.write(row, col,     key)
#     for item in d[key]:
#         worksheet.write(row, col + 1, item)
#         row += 1
#
# workbook.close()









# with open(output_csv_file_name, 'wb') as output_file:
#     for necktype_result in [crewneck_results,
#                             roundneck_results,
#                             scoopneck_results,
#                             squareneck_results,
#                             vneck_results]:
#         keys = necktype_result[0].keys()
#         dict_writer = csv.DictWriter(output_file, keys)
#         dict_writer.writeheader()
#         dict_writer.writerows(necktype_result)

