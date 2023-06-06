import numpy as np
import imageio.v2 as imageio
import glob, argparse, sys
sys.path.append('images')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

### Extract color histogram for each image
def color_histogram(image):
    # Convert the image into gray scale. 
 	# Traverse through the pixels and add to the histogram. 
	# there's 256 bins, one for each pixel
	hist_vector = np.zeros((256,))
	gray_image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
	for i in range(gray_image.shape[0]):
		for j in range (gray_image.shape[1]):
			hist_vector[int(gray_image[i][j])] += 1
	# normalize the histogram
	hist_vector /= sum(hist_vector)
	return list(hist_vector)



### Extract LBP histogram for each image
def lbp_histogram(image):
	# Helper function that determines if the neighboring pixel is bigger that the current,
	# and if it is we return 1. 
 	# Else, return 0.
	def get_pixel(img, center, x, y):
		pixel = 0
		if img[x][y] >= center:
			pixel = 1
		return pixel

	# Helper function that takes in the current pixels position and looks at all neighboring pixels and finds
	# their value with the assistance of get_pixel(), and then returns the sum of all of the values to find 
	# the lbp
	def lbp(x, y, gray_image):
		center = gray_image[x][y]
		lbp_val = []
		lbp_val.append(get_pixel(gray_image, center, x-1, y-1) * 1)
		lbp_val.append(get_pixel(gray_image, center, x-1, y) * 2)
		lbp_val.append(get_pixel(gray_image, center, x-1, y + 1) * 4)
		lbp_val.append(get_pixel(gray_image, center, x, y + 1) * 8)
		lbp_val.append(get_pixel(gray_image, center, x + 1, y + 1) * 16)
		lbp_val.append(get_pixel(gray_image, center, x + 1, y) * 32)
		lbp_val.append(get_pixel(gray_image, center, x + 1, y-1) * 64)
		lbp_val.append(get_pixel(gray_image, center, x, y-1) * 128)
		return sum(lbp_val)

	# convert RGB image to grayscale
	gray_image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

	# Access pixel (h,w) of gray image
	hist_vector = np.zeros((256,))
	for i in range (1, gray_image.shape[0]-1):
		for j in range (1, gray_image.shape[1]-1):
			hist_vector[lbp(i, j, gray_image)] += 1

	# normalize the histogram
	hist_vector /= sum(hist_vector)
	return list(hist_vector)

def find_boundary(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    drawing = False
    rect = (0, 0, 0, 0)
    cv2.namedWindow('Please drag your mouse around the tail')
    cv2.imshow('Please drag your mouse around the tail', image)
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect = (x, y, 0, 0)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = (rect[0], rect[1], x - rect[0], y - rect[1])
            # Calculate width and height
            rect = (rect[0], rect[1], rect[2], rect[3])  
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
            cv2.imshow('Please drag your mouse around the tail', image)
            background = np.zeros((1, 65), np.float64)
            front = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, mask, rect, background, front, 1, cv2.GC_INIT_WITH_RECT)
            # Wait before closing the window
            cv2.waitKey(1)  
            cv2.destroyWindow('Please drag your mouse around the tail')
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                copy = image.copy()
                rect = (rect[0], rect[1], x - rect[0], y - rect[1])  
                rect = (rect[0], rect[1], rect[2], rect[3])  
                cv2.rectangle(copy, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
                cv2.imshow('Please drag your mouse around the tail', copy)
    cv2.setMouseCallback('Please drag your mouse around the tail', mouse_callback)
    cv2.waitKey(0)
    mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    masked_image = cv2.bitwise_and(image, image, mask=mask_fg)
    cv2.destroyAllWindows()
    return masked_image, rect



# Removes the background and only keeps whats inside the boundary boxes. Also resizes them so they're all accurate
def remove_background(image, rect):
    new_image = resize_image(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], 300, 300)
    mask = np.zeros_like(new_image[:, :, 0])
    mask[new_image[:, :, 0] != 0] = 255
    new_image = cv2.bitwise_and(new_image, new_image, mask=mask)
    background = np.ones_like(new_image) * 255
    result = cv2.subtract(background, new_image)
    return result

# function that resizes images with a given width and height
def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# calculates the feature vector for the image
def calculate_feature(image, rect):
    # Remove background from the image
    current_image = remove_background(image, rect)
    # create and return the feature vector as a list
    feature_vector = []
    feature_vector += color_histogram(current_image)
    feature_vector += lbp_histogram(current_image)
    return feature_vector

# creates and prints out the progression bar to the console
def progression(iter, total, prefix='', suffix='', fill='█'):
    percent = ("{0:." + str(1) + "f}").format(100 * (iter / float(total)))
    bar_length = int(50 * iter / total)
    bar = fill * bar_length + '-' * (50 - bar_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

########## MAIN PROGRAM ##########


if __name__ == "__main__":

	### Provide the name of the query image
	ap = argparse.ArgumentParser()
	ap.add_argument("-q", "--query", type=str, required=True, help="name of the query image")
	args = ap.parse_args()

	### Get all the image names in the database
	image_names = sorted(glob.glob('images\\*.jpg'))
	num_images = len(image_names)
	features = []
	rectangle = {'images\\dolphin_1.jpg': (86, 43, 287, 307), 'images\\dolphin_2.jpg': (7, 10, 299, 409), 'images\\dolphin_3.jpg': (56, 21, 344, 243), 'images\\dolphin_4.jpg': (27, 112, 413, 241), 'images\\dolphin_5.jpg': (24, 71, 430, 352), 'images\\shark_1.jpg': (113, 88, 221, 194), 'images\\shark_2.jpg': (34, 30, 299, 362), 'images\\shark_3.jpg': (172, 78, 223, 200), 'images\\shark_4.jpg': (133, 38, 254, 220), 'images\\shark_5.jpg': (171, 112, 170, 167), 'images\\whale_1.jpg': (87, 44, 257, 201), 'images\\whale_2.jpg': (93, 93, 321, 191), 'images\\whale_4.jpg': (6, 47, 438, 275), 'images\\whale_5.jpg': (106, 143, 221, 150), 'images\\whale_3.jpg': (91, 53, 261, 222)}
	query_name = 'images\\' + args.query + '.jpg'
	### Loop over each image and extract a feature vector
	for name in image_names:
		if name == query_name:
			continue
		print('Extracting features from the dataset')
		image = imageio.imread(name)
		image = resize_image(image, 456, 456)
		image = np.clip(image * 1.2, 0, 255).astype(np.uint8)
		feature = calculate_feature(image, rectangle[name])
		features.append(feature)

	### Read the query image and extract its feature vector
	query_image = imageio.imread(query_name)
	query_image = resize_image(query_image, 456, 456)
	query_image = np.clip(query_image * 1.2, 0, 255).astype(np.uint8)

	### Perform interactive foreground masking on the query image
	image, rect = find_boundary(query_image)

	### Extract the feature vector from the masked query image
	query_feature = calculate_feature(query_image, rect)
	#features.append(query_feature)
	### Compare the query feature with the database features
	query_feature = np.reshape(np.array(query_feature), (1, len(query_feature)))
	features = np.array(features)
	distances = cdist(query_feature, features, 'euclidean')
	rectangle[query_name] = rect

	### Sort the distance values in ascending order
	distances = list(distances[0, :])
	sorted_distances = sorted(distances)
	sorted_imagenames = []


	### Perform retrieval; plot the images and save the result as an image file in the working folder
	fig = plt.figure()
	for i in range(num_images-1):
		progression(i + 1, num_images-1, prefix='Progress:', suffix='Complete')
		fig.add_subplot(5, 8, i + 1)
		image_name = image_names[distances.index(sorted_distances[i])]
		sorted_imagenames.append(image_name.split('\\')[-1].rstrip('.jpg'))
		image1 = resize_image(imageio.imread(image_name), 456, 456)
		image1 = np.clip(image1 * 1.2, 0, 255).astype(np.uint8)
		plt.imshow(remove_background(image1, rectangle[image_name]))
		plt.axis('off')
		plt.title(sorted_imagenames[i].split('_')[0].split('\\')[-1])

	figure_save_name = 'Q_' + args.query + '.png'
	plt.savefig(args.query, bbox_inches='tight')
	plt.close(fig)

	### Calculate and print precision value (in percentage)
	precision = 0
	query_class = args.query.split('_')[0]
	d, w = 0, 0
	for i in range(5):
		retrieved_class = sorted_imagenames[i].split('_')[0].split('\\')[-1]
		if retrieved_class == query_class:
			precision += 1
		if retrieved_class == 'dolphin':
			d+=1
		elif retrieved_class == 'whale':
			w +=1
	print()
	if(d >= 3):
		print("The given feature in the query belongs to a dolphin")
	else:
		print("The given feature in the query belongs to a whale")
