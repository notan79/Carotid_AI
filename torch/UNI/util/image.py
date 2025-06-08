import numpy as np

# Pad an image
def pad(img: np.array)->np.array:
    # Tile must be 224 x 224 x 3
    IMG_DIM = 224
    
    print(f'Original shape: {img.shape}')
    
    h, w = img.shape[:2]

    pad_h = (IMG_DIM - h % IMG_DIM) % IMG_DIM
    pad_w = (IMG_DIM - w % IMG_DIM) % IMG_DIM

    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
    print(f'New shape: {img.shape}')
    return img
    
# Return big image as an array of 224x224x3 patches
def get_patches(img: np.array) -> tuple:
    IMG_DIM = 224
    
    img = pad(img)
    
    # Get h, w
    h, w = img.shape[:2]

    num_h = h // IMG_DIM
    num_w = w // IMG_DIM

    # Get the images into patches
    reshaped = img.reshape(num_h, IMG_DIM, num_w, IMG_DIM, 3)
    patches = reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, IMG_DIM, IMG_DIM, 3)
    return patches, num_w, num_h
    