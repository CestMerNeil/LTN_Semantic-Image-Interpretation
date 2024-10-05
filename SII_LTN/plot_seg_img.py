import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def resize_image(image, max_size=65535):
    """
    Resize image if it exceeds a certain maximum size to avoid rendering issues.
    
    Args:
    - image: The input image (numpy array).
    - max_size: The maximum allowed size for the image (default: 65535 pixels).
    
    Returns:
    - Resized image if the original image exceeds the max size.
    """
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scaling_factor = min(max_size / height, max_size / width)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size)
        return resized_image
    return image

def plot_seg(results, show_orig_image=True, show_bboxes=False, alpha=0.5):
    """
    Display YOLO segmentation results with options for showing original image,
    bounding boxes, and controlling mask transparency. Different classes will 
    be displayed in different colors, with class labels added to the segmentation regions.
    
    Args:
    - results: YOLO inference result object
    - show_orig_image: Whether to show the original image (default: True)
    - show_bboxes: Whether to show bounding boxes (default: False)
    - alpha: Transparency for the mask overlay (default: 0.5)
    """
    
    # Get the first result
    result = results[0]
    
    # Original image and segmentation masks
    image_with_masks = result.orig_img
    image_with_masks = cv2.cvtColor(image_with_masks, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Resize the image to avoid exceeding size limits
    image_with_masks = resize_image(image_with_masks)  # Resize if too large
    
    masks = result.masks
    classes = result.boxes.cls.cpu().numpy()  # Get detected classes (converted to numpy)
    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
    class_names = result.names  # Get class names
    
    # Create a combined mask for overlay
    combined_mask = np.zeros_like(image_with_masks, dtype=np.uint8)
    
    # Assign random colors to each class
    unique_classes = np.unique(classes)
    class_colors = {cls: [random.randint(0, 255) for _ in range(3)] for cls in unique_classes}
    
    # Process masks if available
    if masks is not None:
        for idx, mask in enumerate(masks.data):
            mask_np = mask.cpu().numpy()  # Convert mask to numpy
            mask_np_resized = cv2.resize(mask_np, (image_with_masks.shape[1], image_with_masks.shape[0]))  # Resize to match image
            class_id = int(classes[idx])  # Get class ID for this mask
            color = class_colors[class_id]  # Get color for this class
            class_name = class_names[class_id]  # Get class name from the names list

            # Apply color to the segmented regions only
            for c in range(3):  # Apply color to each channel (R, G, B)
                combined_mask[:, :, c] = np.where(mask_np_resized > 0.5, color[c], combined_mask[:, :, c])

            # Add class labels to the center of the segmented area
            y_center, x_center = np.mean(np.where(mask_np_resized > 0.5), axis=1).astype(int)
            label = class_name  # Use class name instead of ID
            plt.text(x_center, y_center, label, color=np.array(color)/255, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
    
    if show_orig_image:
        # Blend original image and mask (use alpha to control transparency)
        blended_image = cv2.addWeighted(image_with_masks, 1, combined_mask, alpha, 0)
        plt.imshow(blended_image)
    else:
        # Show only the segmentation mask
        plt.imshow(combined_mask)

    # Optionally draw bounding boxes
    if show_bboxes:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[idx])  # Get class ID for this bounding box
            color = class_colors[class_id]  # Get the color for this class
            class_name = class_names[class_id]  # Get class name from the names list
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                              fill=False, edgecolor=np.array(color)/255, linewidth=2))
            # Add label near the bounding box
            plt.text(x1, y1 - 10, class_name, color=np.array(color)/255, fontsize=12, 
                     bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.show()
