import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define function to plot image with bounding box in yolo format
def plot_img_with_bbox_yolo(image, label_list, class_labels_dict=None, save_path=None):
    """
    Plot the given image with a bounding box in yolo format.
    yolo format: <class> <center_x> <center_y> <width> <height> (Normalised by image width and height)

    Parameters:
    image (numpy.ndarray): The input image to be plotted.
    label_list: The bounding box in yolo format.
    class_labels_dict: The class labels dictionary. 
    Returns:
    None 
    """

    # Plot the image
    plt.imshow(image)
    ax = plt.gca()
    for label in label_list:
        c, center_x, center_y, w, h = label.split(' ')
        # Convert string to float
        center_x, center_y, w, h = float(center_x), float(center_y), float(w), float(h)
        # Convert center_x, center_y, w, h to top, left, width, height
        w = w * image.shape[1]
        h = h * image.shape[0]
        center_x = center_x * image.shape[1]
        center_y = center_y * image.shape[0]
        # Convert center_x, center_y to left, bottom
        left = center_x - w/2
        bottom = center_y - h/2
       
        # Create a Rectangle patch
        rect = patches.Rectangle((left, bottom), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)
        # Add the class label to the plot
        if class_labels_dict:
            ax.text(left, bottom, class_labels_dict[int(c)], color='r')   
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

project_path = './' # workstation
mask_path = project_path + '/data/masks/' # define path to MT labels


# get list of masks from the folder
masks_filenames = os.listdir(mask_path)

# load a mask image
i = 1
mask_image_path = os.path.join(mask_path, masks_filenames[i])
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if mask_image is None:
    raise FileNotFoundError(f"Image not found at path: {mask_image_path}")

# Print the shape of the mask image
print(mask_image.shape)

# Print the maximum value in the mask image
num_masks_in_img = int(mask_image.max())

# plot the mask image
plt.imshow(mask_image, cmap='gray')
plt.axis('on')
plt.show()

# get bounding box of the mask
contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Print the number of contours found in the mask image
print(f"Number of contours found: {len(contours)}")

bboxes = []
# get bounding boxes of the contours
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
    bboxes.append((x, y, w, h))

# plot mask with bounding boxes
mask_with_bboxes = mask_image.copy()
for bbox in bboxes:
    x, y, w, h = bbox
    cv2.rectangle(mask_with_bboxes, (x, y), (x + w, y + h), (num_masks_in_img+1, 0, 0), 2)

plt.imshow(mask_with_bboxes, cmap='gray')
plt.axis('on')
plt.show()

# get labels in yolo format to write in a text file ...

label_list = []
for box in bboxes:
    bbox_x, bbox_y, bbox_w, bbox_h = box
    # print('Bounding box: x={}, y={}, w={}, h={}'.format(bbox_x, bbox_y, bbox_w, bbox_h))

    # convert boundingbox to coco format (top, left, w, h) -> center_x, center_y, w, h (normalised by image width and height) 
    coco_bbox = [bbox_x + bbox_w/2, bbox_y + bbox_h/2, bbox_w, bbox_h]
    # normalise by image width and height
    coco_bbox[0] = coco_bbox[0] / mask_image.shape[1]
    coco_bbox[1] = coco_bbox[1] / mask_image.shape[0]
    coco_bbox[2] = coco_bbox[2] / mask_image.shape[1]
    coco_bbox[3] = coco_bbox[3] / mask_image.shape[0]
    # print('COCO bounding box: ' + str(coco_bbox))
    
    # create bounding box label in txt format ie. <class> <center_x> <center_y> <width> <height>
    label_list.append('0 ' + ' '.join([str(i) for i in coco_bbox]))

# print the label list
print("Label list in yolo format:")
print(label_list)

# plot the image with bounding boxes
plot_img_with_bbox_yolo(mask_image, label_list, class_labels_dict=None, save_path=None)
