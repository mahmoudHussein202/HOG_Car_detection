from distutils.log import debug
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import random
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
import time
from sklearn.preprocessing import _data
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import queue
from scipy.ndimage.measurements import label
import os
import pickle
import funct
import classes

def find_vehicles(image):
    bboxes = funct.get_rectangles(image,svc , X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins) 
    detection_info.add_bboxes(bboxes)
    labels = detection_info.get_labels()
    if len(labels) == 0:
        result_image = image
    else:
        bboxes, result_image = funct.draw_labeled_bboxes(image,labels)

    return result_image
#------------------------main-----------------------------    


debug = 0
mode=0
vehicles_dir =     'data\\vehicles'
non_vehicles_dir = 'data\\non-vehicles'
#Read cars and not-cars images
#Data folders
# images are divided up into vehicles and non-vehicles
cars = []
notcars = []
# Read vehicle images
images = glob.iglob(vehicles_dir + '/**/*.png', recursive=True)
for image in images:
        cars.append(image)
# Read non-vehicle images
images = glob.iglob(non_vehicles_dir + '/*/*.png', recursive=True)
for image in images:
        notcars.append(image)
data_info = funct.data_look(cars, notcars)

#---------------------------------------------------------------


num_images = 10

# Just for fun choose random car / not-car indices and plot example images   
cars_samples = random.sample(list(cars), num_images)
notcar_samples = random.sample(list(notcars), num_images)
    
# Read in car / not-car images
car_images = []
notcar_images = []
for sample in cars_samples:
    car_images.append(mpimg.imread(sample))
    
for sample in notcar_samples:
    notcar_images.append(mpimg.imread(sample))


#----------------------------------------------------------

orient = 9
pix_per_cell = 8
cell_per_block = 2

car_features, hog_image = funct.get_hog_features(cv2.cvtColor(car_images[1], cv2.COLOR_RGB2GRAY), orient, pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=True)

notcar_features, notcar_hog_image = funct.get_hog_features(cv2.cvtColor(notcar_images[2], cv2.COLOR_RGB2GRAY), orient, pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=True)

#-----------------------------train-------------------------------
if mode ==0 :

    print("Training the model, please wait...")

    colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size=(32, 32)
    hist_bins=32

    t=time.time()

    car_features = funct.extract_features(cars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, hist_bins=hist_bins)
    notcar_features = funct.extract_features(notcars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, hist_bins=hist_bins)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64) 
    print(X.shape)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    print(len(y))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.15, random_state=rand_state)

    #X_train, X_test = shuffle(X_train, y_train, random_state=rand_state)
    # # Compute a PCA  on the features 
    # n_components = 4932

    # print("Extracting the top %d features from %d total features"
    #       % (n_components, X_train.shape[1]))

    # pca = PCA(n_components=n_components, svd_solver='randomized',
    #           whiten=True).fit(X_train)

    # X_train_pca = pca.transform(X_train)
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC X_scaler
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
#---------------------------------------------------------------------------

#--------------------save_training_model-------------------------

#Pickle the data as it takes a lot of time to generate it
    data_file = 'results\\model\\svc_pickle.p'

    if not os.path.isfile(data_file):
        with open(data_file, 'wb') as pfile:
            pickle.dump(
                {
                    'svc': svc,
                    'scaler': X_scaler,
                    'orient': orient,
                    'pix_per_cell': pix_per_cell,
                    'cell_per_block': cell_per_block,
                    'spatial_size': spatial_size,
                    'hist_bins': hist_bins
                    
                },
                pfile, pickle.HIGHEST_PROTOCOL)

    print('Data saved in pickle file')
elif mode ==1 :
    pickle_file = open('results\\model\\svc_pickle.p','rb')
    data = pickle.load(pickle_file)
    svc=0
    X_scaler=0
    spatial_size=0
    hist_bins=0
    orient=0
    pix_per_cell=0

#----------------------------Read cars and not-cars images--------------------------------
#Data folders
# images are divided up into vehicles and non-vehicles
#Read cars and not-cars images
#Data folders
# images are divided up into vehicles and non-vehicles
test_images = []
images = glob.glob('data/test_images/*.jpg')
for image in images:
        test_images.append(mpimg.imread(image))
result_images = []
result_boxes = []
heatmap_images = []
result_img_all_boxes = []
for test_image in test_images:
    rectangles = funct.get_rectangles(test_image,svc , X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    result_img_all_boxes.append(funct.draw_boxes(test_image, rectangles, color='random', thick=3))
    heatmap_image = np.zeros_like(test_image[:, :, 0])
    heatmap_image = funct.add_heat(heatmap_image, rectangles)
    heatmap_images.append(heatmap_image)
    heatmap_image = funct.apply_threshold(heatmap_image, 2)
    labels = label(heatmap_image)
    rectangles, result_image = funct.draw_labeled_bboxes(test_image, labels)
    result_boxes.append(rectangles)
    result_images.append(result_image)



src = input("Enter path of the video or image :")
src= str(src)
#------------------------Video Write---------------------------

if src[len(src)-3:] == 'mp4':
    detection_info = classes.DetectionInfo(test_images)
    detection_info.old_heatmap = np.zeros_like(test_images[0][:, :, 0])
    input_name= input("Enter a name for the output Video :")
    project_video_path = src
    project_video_output = 'results\\output_videos\\'+input_name+'.mp4'
    print("output will be in results\output_videos folder")
    project_video = VideoFileClip(project_video_path)
    white_clip = project_video.fl_image(find_vehicles) 
    white_clip.write_videofile(project_video_output, audio=False)
#--------------------------------------------------------------
else:
    result_images = []
    result_boxes = []
    heatmap_images = []
    result_img_all_boxes = []
    test_image = cv2.imread(src)
    rectangles = funct.get_rectangles(test_images,svc , X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins) 
    result_img_all_boxes.append(funct.draw_boxes(test_image, rectangles, color='random', thick=3))
    heatmap_image = np.zeros_like(test_image[:, :, 0])
    heatmap_image = funct.add_heat(heatmap_image, rectangles)
    heatmap_images.append(heatmap_image)
    heatmap_image = funct.apply_threshold(heatmap_image, 2)
    labels = label(heatmap_image)
    rectangles, result_image = funct.draw_labeled_bboxes(test_image, labels)
    result_boxes.append(rectangles)
    result_images.append(result_image)
    funct.visualize_images(result_img_all_boxes, 2, "test")


