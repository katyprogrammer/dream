# helper functions
from IPython.display import Image, display
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

# Image manipulation.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
# import inception5h
# inception5h.data_dir = 'inception/5h/'
# inception5h.maybe_download()
# model 
def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)

def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Convert to bytes.
    image = image.astype(np.uint8)
    
    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')
        
def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.
    
    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image/255.0, 0.0, 1.0)
        
        # Plot using matplotlib.
        # plt.imshow(image, interpolation='lanczos')
        # plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)
        
        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        display(PIL.Image.fromarray(image))
        
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)
    
    return x_norm

def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)
    
    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()
    
def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor
        
        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]
    
    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)
    
    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)
    
    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)
    
    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)
    
    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized

def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))
    
    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)
    
    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)
    
    return actual_tile_size

def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)

    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size
        
        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g
            
            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad

def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.
    
    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """

    # Copy the image so we don't overwrite the original image.
    img = image.copy()
    
    # print("Image before:")
    # plot_image(img)

    print("Processing image: ", end="")

    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.
    gradient = model.get_gradient(layer_tensor)
    
    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img)
        
        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    print()
    # print("Image after:")
    # plot_image(img)
    
    return img

def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.

    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """

    # Do a recursive step?
    if num_repeats>0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)
            
        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)
        
        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)
    
    return img_result
    
import numpy as np
import pickle
import keras.backend as K
def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    print(K.image_dim_ordering())
    if K.image_dim_ordering() == 'tf':    
        images = images.transpose([0, 2, 3, 1])
    return images

def onehot_labels(labels):
    return np.eye(100)[labels]

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def load_class_names():
    """
    Unpickle the given file and return the data.
    extract label names from the data

    """

    # Create full path for the file.
    file_path = 'data/cifar-100-python/meta'

    print("Loading class label: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    # Load the class-names from the pickled file.
    # @TODO what's the name of the label name is the meta file
    raw = data[b'fine_label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]
    
    return names

def load_coarse_class_names():
    """
    Unpickle the given file and return the data.
    extract label names from the data

    """

    # Create full path for the file.
    file_path = 'data/cifar-100-python/meta'

    print("Loading class label: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    # Load the class-names from the pickled file.
    # @TODO what's the name of the label name is the meta file
    raw = data[b'coarse_label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]
    
    return names

X_train = get_proper_images(unpickle('data/cifar-100-python/train')[b'data'])
Y_train = onehot_labels(unpickle('data/cifar-100-python/train')[b'fine_labels'])
X_test = get_proper_images(unpickle('data/cifar-100-python/test')[b'data'])
Y_test = onehot_labels(unpickle('data/cifar-100-python/test')[b'fine_labels'])
print(X_train.shape)
Y_test.shape

# from matplotlib import pyplot as plt
from scipy.misc import toimage
from keras import backend as K
import numpy as np
# load data

# def sample_images(X_train):
#     # create a grid of 3x3 images
#     start = int(np.random.rand()*2500)-10
#     for i in range(0, 9):
#         plt.subplot(3,3,1 + i)
#         plt.imshow(toimage(X_train[i+start]))
#         plt.axis('off')
#     # show the plot
#     plt.show()

cls_names = load_class_names()
# cls_names

cls_coarse_names = load_coarse_class_names()
# cls_coarse_names

X_train = get_proper_images(unpickle('data/cifar-100-python/train')[b'data'])
Y_train = onehot_labels(unpickle('data/cifar-100-python/train')[b'fine_labels'])
Y_train_class = unpickle('data/cifar-100-python/train')[b'fine_labels']
Y_train_coarse = unpickle('data/cifar-100-python/train')[b'coarse_labels']
X_test = get_proper_images(unpickle('data/cifar-100-python/test')[b'data'])
Y_test = onehot_labels(unpickle('data/cifar-100-python/test')[b'fine_labels'])

# Y_test_class is the single number version of Y_test ex: [0, 0, 1] -> 2
Y_test_class = unpickle('data/cifar-100-python/test')[b'fine_labels']
Y_test_coarse = unpickle('data/cifar-100-python/test')[b'coarse_labels']
Y_test.shape

X_train_people = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==14]
Y_train_people = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==14]

X_train_small_mammals = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==16]
Y_train_small_mammals = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==16]

X_train_medium_sized_mammals = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==12]
Y_train_medium_sized_mammals = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==12]

X_train_aquatic_mammals = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==0]
Y_train_aquatic_mammals = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==0]

X_train_fish = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==1]
Y_train_fish = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==1]

X_train_reptiles = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==15]
Y_train_reptiles = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==15]

X_train_carnivores = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==8]
Y_train_carnivores = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label==8]

# test data
X_test_people = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==14]
Y_test_people = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==14]

X_test_small_mammals = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==16]
Y_test_small_mammals = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==16]

X_test_medium_sized_mammals = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==12]
Y_test_medium_sized_mammals = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==12]

X_test_aquatic_mammals = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==0]
Y_test_aquatic_mammals = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==0]

X_test_fish = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==1]
Y_test_fish = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==1]

X_test_reptiles = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==15]
Y_test_reptiles = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==15]

X_test_carnivores = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==8]
Y_test_carnivores = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==8]

# combines the five dataset
X_train_five = np.concatenate((X_train_people,X_train_small_mammals,X_train_medium_sized_mammals,X_train_aquatic_mammals,X_train_fish), axis=0)
Y_train_five = np.concatenate((Y_train_people,Y_train_small_mammals,Y_train_medium_sized_mammals,Y_train_aquatic_mammals,Y_train_fish), axis=0)
X_test_five = np.concatenate((X_test_people,X_test_small_mammals,X_test_medium_sized_mammals,X_test_aquatic_mammals,X_test_fish), axis=0)
Y_test_five = np.concatenate((Y_test_people,Y_test_small_mammals,Y_test_medium_sized_mammals,Y_test_aquatic_mammals,Y_test_fish), axis=0)

# convert type
X_train_people = np.asarray(X_train_people)
X_train_small_mammals = np.asarray(X_train_small_mammals)
X_train_medium_sized_mammals = np.asarray(X_train_medium_sized_mammals)
X_train_aquatic_mammals = np.asarray(X_train_aquatic_mammals)
X_train_fish = np.asarray(X_train_fish)
X_train_five = np.asarray(X_train_five)
X_train_reptiles = np.asarray(X_train_reptiles)
X_train_carnivores = np.asarray(X_train_carnivores)


X_test_people = np.asarray(X_test_people)
X_test_small_mammals = np.asarray(X_test_small_mammals)
X_test_medium_sized_mammals = np.asarray(X_test_medium_sized_mammals)
X_test_aquatic_mammals = np.asarray(X_test_aquatic_mammals)
X_test_fish = np.asarray(X_test_fish)
X_test_five = np.asarray(X_test_five)
X_test_reptiles = np.asarray(X_test_reptiles)
X_test_carnivores = np.asarray(X_test_carnivores)

Y_train_people = np.asarray(Y_train_people)
Y_train_small_mammals = np.asarray(Y_train_small_mammals)
Y_train_medium_sized_mammals = np.asarray(Y_train_medium_sized_mammals)
Y_train_aquatic_mammals = np.asarray(Y_train_aquatic_mammals)
Y_train_fish = np.asarray(Y_train_fish)
Y_train_five = np.asarray(Y_train_five)
Y_train_reptiles = np.asarray(Y_train_reptiles)
Y_train_carnivores = np.asarray(Y_train_carnivores)

Y_test_people = np.asarray(Y_test_people)
Y_test_small_mammals = np.asarray(Y_test_small_mammals)
Y_test_medium_sized_mammals = np.asarray(Y_test_medium_sized_mammals)
Y_test_aquatic_mammals = np.asarray(Y_test_aquatic_mammals)
Y_test_fish = np.asarray(Y_test_fish)
Y_test_five = np.asarray(Y_test_five)
Y_test_reptiles = np.asarray(Y_test_reptiles)
Y_test_carnivores = np.asarray(Y_test_carnivores)

# another way to split data with more data per episode
X_test_1 = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 0 and Y_label <= 3]
Y_test_1 = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 0 and Y_label <= 3]

X_test_2 = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 4 and Y_label <= 7]
Y_test_2 = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 4 and Y_label <= 7]

X_test_3 = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 8 and Y_label <= 11]
Y_test_3 = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 8 and Y_label <= 11]

X_test_4 = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 12 and Y_label <= 15]
Y_test_4 = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 12 and Y_label <= 15]

X_test_5 = [X_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 16 and Y_label <= 19]
Y_test_5 = [Y_test[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label >= 16 and Y_label <= 19]


X_train_1 = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 0 and Y_label <= 3]
Y_train_1 = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 0 and Y_label <= 3]

X_train_2 = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 4 and Y_label <= 7]
Y_train_2 = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 4 and Y_label <= 7]

X_train_3 = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 8 and Y_label <= 11]
Y_train_3 = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 8 and Y_label <= 11]

X_train_4 = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 12 and Y_label <= 15]
Y_train_4 = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 12 and Y_label <= 15]

X_train_5 = [X_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 16 and Y_label <= 19]
Y_train_5 = [Y_train[i] for i, Y_label in enumerate(Y_train_coarse) if Y_label >= 16 and Y_label <= 19]

X_train_1 = np.asarray(X_train_1)
X_train_2 = np.asarray(X_train_2)
X_train_3 = np.asarray(X_train_3)
X_train_4 = np.asarray(X_train_4)
X_train_5 = np.asarray(X_train_5)

X_test_1 = np.asarray(X_test_1)
X_test_2 = np.asarray(X_test_2)
X_test_3 = np.asarray(X_test_3)
X_test_4 = np.asarray(X_test_4)
X_test_5 = np.asarray(X_test_5)

Y_train_1 = np.asarray(Y_train_1)
Y_train_2 = np.asarray(Y_train_2)
Y_train_3 = np.asarray(Y_train_3)
Y_train_4 = np.asarray(Y_train_4)
Y_train_5 = np.asarray(Y_train_5)

Y_test_1 = np.asarray(Y_test_1)
Y_test_2 = np.asarray(Y_test_2)
Y_test_3 = np.asarray(Y_test_3)
Y_test_4 = np.asarray(Y_test_4)
Y_test_5 = np.asarray(Y_test_5)

X_train_easy = [X_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==0]
X_train_easy = np.concatenate((X_train_easy, [X_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==20]), axis=0)
X_train_easy = np.concatenate((X_train_easy, [X_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==40]), axis=0)
X_train_easy = np.concatenate((X_train_easy, [X_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==60]), axis=0)
X_train_easy = np.concatenate((X_train_easy, [X_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==80]), axis=0)
X_train_easy = np.asarray(X_train_easy)

Y_train_easy = [Y_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==0]
Y_train_easy = np.concatenate((Y_train_easy, [Y_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==20]), axis=0)
Y_train_easy = np.concatenate((Y_train_easy, [Y_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==40]), axis=0)
Y_train_easy = np.concatenate((Y_train_easy, [Y_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==60]), axis=0)
Y_train_easy = np.concatenate((Y_train_easy, [Y_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==80]), axis=0)
Y_train_easy = np.asarray(Y_train_easy)

X_test_easy = [X_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==0]
X_test_easy = np.concatenate((X_test_easy, [X_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==20]), axis=0)
X_test_easy = np.concatenate((X_test_easy, [X_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==40]), axis=0)
X_test_easy = np.concatenate((X_test_easy, [X_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==60]), axis=0)
X_test_easy = np.concatenate((X_test_easy, [X_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==80]), axis=0)
X_test_easy = np.asarray(X_test_easy)

Y_test_easy = [Y_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==0]
Y_test_easy = np.concatenate((Y_test_easy, [Y_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==20]), axis=0)
Y_test_easy = np.concatenate((Y_test_easy, [Y_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==40]), axis=0)
Y_test_easy = np.concatenate((Y_test_easy, [Y_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==60]), axis=0)
Y_test_easy = np.concatenate((Y_test_easy, [Y_test[i] for i, Y_label in enumerate(Y_test_class) if Y_label==80]), axis=0)
Y_test_easy = np.asarray(Y_test_easy)

print(X_train_five.shape)
print(X_train_easy.shape)
print(X_train_people.shape)
Y = [Y_train[i] for i, Y_label in enumerate(Y_train_class) if Y_label==20]
len(Y)

Y_test_class_people = [Y_test_class[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==14]
Y_test_class_small_mammals = [Y_test_class[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==16]
Y_test_class_medium_sized_mammals = [Y_test_class[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==12]
Y_test_class_aquatic_mammals = [Y_test_class[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==0]
Y_test_class_fish = [Y_test_class[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==1]
Y_test_class_reptiles = [Y_test_class[i] for i, Y_label in enumerate(Y_test_coarse) if Y_label==15]
Y_test_class_five = np.concatenate((Y_test_class_people, Y_test_class_small_mammals, Y_test_class_medium_sized_mammals, Y_test_class_aquatic_mammals, Y_test_class_fish), axis=0)

                                    
Y_test_class_people = np.asarray(Y_test_class_people)
Y_test_class_small_mammals = np.asarray(Y_test_class_small_mammals)
Y_test_class_medium_sized_mammals = np.asarray(Y_test_class_medium_sized_mammals)
Y_test_class_aquatic_mammals = np.asarray(Y_test_class_aquatic_mammals)
Y_test_class_fish = np.asarray(Y_test_class_fish)
Y_test_class_reptilse = np.asarray(Y_test_class_reptiles)
Y_test_class_five = np.asarray(Y_test_class_five)


from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0)

import keras.backend as K
K._LEARNING_PHASE = tf.constant(1)

from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras import applications
from os.path import expanduser
import os

def load_model_whole():
    # HOME = expanduser("~")
    # MODEL_PATH = os.path.join(HOME, 'katy/good_model.h5')
    # vgg16 = load_model(MODEL_PATH)
    vgg16 = load_model('good_model.h5')

    top = vgg16.get_layer('dropout_56').output
    # print(top)
    prediction = Dense(output_dim=100, activation='softmax', name='softmax', input_shape=(512, None, None))(top)
    model_whole = Model(input=vgg16.input, output=prediction)
    print(vgg16.input.shape)
    # model_whole.summary()
    print("datalab pretrained model loaded")
    return model_whole

# load the variance list

with open('variance/X_train_people_var.txt', 'rb') as f:
    X_train_people_var = pickle.load(f)
X_train_people_var= np.asarray(X_train_people_var)
with open('variance/X_train_small_mammals_var.txt', 'rb') as f:
    X_train_small_mammals_var = pickle.load(f)
X_train_small_mammals_var= np.asarray(X_train_small_mammals_var)
with open('variance/X_train_medium_sized_mammals_var.txt', 'rb') as f:
    X_train_medium_sized_mammals_var = pickle.load(f)
X_train_medium_sized_mammals_var = np.asarray(X_train_medium_sized_mammals_var)
with open('variance/X_train_aquatic_mammals_var.txt', 'rb') as f:
    X_train_aquatic_mammals_var = pickle.load(f)
X_train_aquatic_mammals_var = np.asarray(X_train_aquatic_mammals_var)
with open('variance/X_train_fish_var.txt', 'rb') as f:
    X_train_fish_var = pickle.load(f)
X_train_fish_var = np.asarray(X_train_fish_var)
with open('variance/X_train_reptiles_var.txt', 'rb') as f:
    X_train_reptiles_var = pickle.load(f)
X_train_reptiles_var = np.asarray(X_train_reptiles_var)
with open('variance/X_train_carnivores_var.txt', 'rb') as f:
    X_train_carnivores_var = pickle.load(f)
X_train_carnivores_var = np.asarray(X_train_carnivores_var)

# dream function

# partialy train on high-condidence image
def dream(model, X_train_var, X_train, Y_train, X_test, Y_test, confidence_rate =0.5):
    X_train_var_tuple = enumerate(X_train_var)
    # samll variance -> high confidence
    X_train_var_tuple_sorted = sorted(X_train_var_tuple, key = lambda x: x[1], reverse=False)
    X_train_var_tuple_sorted_with_high_confidence_ = X_train_var_tuple_sorted[: int(confidence_rate * len(X_train))]
    X_train_var_index_sorted_with_high_confidence = [x[0] for x in X_train_var_tuple_sorted_with_high_confidence_]
    X_train_high_confidence = [X_train_small_mammals[i] for i in X_train_var_index_sorted_with_high_confidence]
    Y_train_high_confidence = [Y_train_small_mammals[i] for i in X_train_var_index_sorted_with_high_confidence]
    print("train with: ")
    print(len(X_train_high_confidence))
    X_train_high_confidence = np.asarray(X_train_high_confidence)
    Y_train_high_confidence = np.asarray(Y_train_high_confidence)
    sgd = SGD(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    batch_size = 32
    his = model.fit(X_train_high_confidence, Y_train_high_confidence, \
              batch_size=batch_size, \
              nb_epoch=nb_epoch, \
              validation_split=0.2, \
              callbacks=[early_stop], \
              verbose=0, \
              shuffle=True)

    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print('\nTest loss: %.3f' % score[0])
    print('Test accuracy: %.3f' % score[1])
    # acc on small mammals: 0.65
    


from keras.optimizers import SGD, Adam
from keras.models import model_from_json
# load json and create model
json_file = open('models/medium_sized_mammals_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_whole = model_from_json(loaded_model_json)
# load weights into new model
model_whole.load_weights("models/medium_sized_mammals_model.h5")
print("Loaded model from disk")
# test on yesterday episode -> totally forget

# zero shot learning
print("zero shot learning")
model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("zero shot leaning, test on B task")
score = model_whole.evaluate(X_test_aquatic_mammals, Y_test_aquatic_mammals, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

print("zero shot leaning, test on A task")
# test on yesterday episode -> totally forget
score = model_whole.evaluate(X_test_medium_sized_mammals, Y_test_medium_sized_mammals, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

# nb_epoch = 30 is too long, causing catastrophic forgetting on A? is this the reason?

# few shot learning on the next episode (500 image) No dream
  
nums_train_images = [1, 5, 10, 15, 20, 100, 200, 300, 400, 500, 1000]        
all_shots_acc_A = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
all_shots_acc_B = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
for i in range(20):
    for num_train_images in nums_train_images:
        # adjust training epoch
        if num_train_images < 20:
            nb_epoch = 6
        else:
            nb_epoch = 50

        # first initialize the model and let in train on the day time task (task A)
        print(str(i))
        print(str(num_train_images) + " shot leaning, day time")
        K.clear_session()
        # load json and create model
        json_file = open('models/medium_sized_mammals_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_whole = model_from_json(loaded_model_json)
        # load weights into new model
        model_whole.load_weights("models/medium_sized_mammals_model.h5")
        #print("Loaded model from disk")

        print(str(num_train_images) + " shot leaning, night time")
        ### dreaming
        # dream(model_whole, X_train_small_mammals_var, X_train_small_mammals, Y_train_small_mammals, X_test_small_mammals, Y_test_small_mammals)
        
        print(str(num_train_images) + " shot leaning training, on second day task")
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)
        model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # network setting
        batch_size = 32
        his = model_whole.fit(X_train_aquatic_mammals[:num_train_images], Y_train_aquatic_mammals[:num_train_images], \
                  batch_size=batch_size, \
                  nb_epoch=nb_epoch, \
                  verbose=0, \
                  shuffle=True)

        print(str(num_train_images) + " shot leaning, test on B task")
        score_B = model_whole.evaluate(X_test_aquatic_mammals, Y_test_aquatic_mammals, verbose=1)
        print('\nTest loss: %.3f' % score_B[0])
        print('Test accuracy: %.3f' % score_B[1])

        print(str(num_train_images) + " shot leaning, test on A task")
        # test on yesterday episode -> totally forget
        score_A = model_whole.evaluate(X_test_medium_sized_mammals, Y_test_medium_sized_mammals, verbose=1)
        print('\nTest loss: %.3f' % score_A[0])
        print('Test accuracy: %.3f' % score_A[1])
        
        all_shots_acc_A[num_train_images].append(score_A[1])
        all_shots_acc_B[num_train_images].append(score_B[1])
        K.clear_session()
        
with open('accuracy/all_shots_acc_A_with_dream_medium_sized_mammals_to_aquatic_mammals_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_A, f)
with open('accuracy/all_shots_acc_B_with_dream_medium_sized_mammals_to_aquatic_mammals_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_B, f)  
    
    
###########

# load json and create model
json_file = open('models/aquatic_mammals_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_whole = model_from_json(loaded_model_json)
# load weights into new model
model_whole.load_weights("models/aquatic_mammals_model.h5")
print("Loaded model from disk")
# test on yesterday episode -> totally forget

# zero shot learning
print("zero shot learning")
model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("zero shot leaning, test on B task")
score = model_whole.evaluate(X_test_fish, Y_test_fish, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

print("zero shot leaning, test on A task")
# test on yesterday episode -> totally forget
score = model_whole.evaluate(X_test_aquatic_mammals, Y_test_aquatic_mammals, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])


all_shots_acc_A = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
all_shots_acc_B = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
for i in range(20):
    for num_train_images in nums_train_images:
        # adjust training epoch
        if num_train_images < 20:
            nb_epoch = 6
        else:
            nb_epoch = 50

        # first initialize the model and let in train on the day time task (task A)
        print(str(i))
        print(str(num_train_images) + " shot leaning, day time")
        K.clear_session()
        # load json and create model
        json_file = open('models/aquatic_mammals_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_whole = model_from_json(loaded_model_json)
        # load weights into new model
        model_whole.load_weights("models/aquatic_mammals_model.h5")
        #print("Loaded model from disk")

        print(str(num_train_images) + " shot leaning, night time")
        ### dreaming
        # dream(model_whole, X_train_small_mammals_var, X_train_small_mammals, Y_train_small_mammals, X_test_small_mammals, Y_test_small_mammals)
        
        print(str(num_train_images) + " shot leaning training, on second day task")
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)
        model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # network setting
        batch_size = 32
        his = model_whole.fit(X_train_fish[:num_train_images], Y_train_fish[:num_train_images], \
                  batch_size=batch_size, \
                  nb_epoch=nb_epoch, \
                  verbose=0, \
                  shuffle=True)

        print(str(num_train_images) + " shot leaning, test on B task")
        score_B = model_whole.evaluate(X_test_fish, Y_test_fish, verbose=1)
        print('\nTest loss: %.3f' % score_B[0])
        print('Test accuracy: %.3f' % score_B[1])

        print(str(num_train_images) + " shot leaning, test on A task")
        # test on yesterday episode -> totally forget
        score_A = model_whole.evaluate(X_test_aquatic_mammals, Y_test_aquatic_mammals, verbose=1)
        print('\nTest loss: %.3f' % score_A[0])
        print('Test accuracy: %.3f' % score_A[1])
        
        all_shots_acc_A[num_train_images].append(score_A[1])
        all_shots_acc_B[num_train_images].append(score_B[1])
        K.clear_session()
        
with open('accuracy/all_shots_acc_A_with_dream_aquatic_mammals_to_fish_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_A, f)
with open('accuracy/all_shots_acc_B_with_dream_aquatic_mammals_to_fish_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_B, f) 
    
    
    
#######

# load json and create model
json_file = open('models/people_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_whole = model_from_json(loaded_model_json)
# load weights into new model
model_whole.load_weights("models/people_model.h5")
print("Loaded model from disk")
# test on yesterday episode -> totally forget

# zero shot learning
print("zero shot learning")
model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("zero shot leaning, test on B task")
score = model_whole.evaluate(X_test_small_mammals, Y_test_small_mammals, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

print("zero shot leaning, test on A task")
# test on yesterday episode -> totally forget
score = model_whole.evaluate(X_test_people, Y_test_people, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

all_shots_acc_A = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
all_shots_acc_B = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
for i in range(20):
    for num_train_images in nums_train_images:
        # adjust training epoch
        if num_train_images < 20:
            nb_epoch = 6
        else:
            nb_epoch = 50

        # first initialize the model and let in train on the day time task (task A)
        print(str(i))
        print(str(num_train_images) + " shot leaning, day time")
        K.clear_session()
        # load json and create model
        json_file = open('models/people_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_whole = model_from_json(loaded_model_json)
        # load weights into new model
        model_whole.load_weights("models/people_model.h5")
        #print("Loaded model from disk")

        print(str(num_train_images) + " shot leaning, night time")
        ### dreaming
        # dream(model_whole, X_train_small_mammals_var, X_train_small_mammals, Y_train_small_mammals, X_test_small_mammals, Y_test_small_mammals)
        
        print(str(num_train_images) + " shot leaning training, on second day task")
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)
        model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # network setting
        batch_size = 32
        his = model_whole.fit(X_train_small_mammals[:num_train_images], Y_train_small_mammals[:num_train_images], \
                  batch_size=batch_size, \
                  nb_epoch=nb_epoch, \
                  verbose=0, \
                  shuffle=True)

        print(str(num_train_images) + " shot leaning, test on B task")
        score_B = model_whole.evaluate(X_test_small_mammals, Y_test_small_mammals, verbose=1)
        print('\nTest loss: %.3f' % score_B[0])
        print('Test accuracy: %.3f' % score_B[1])

        print(str(num_train_images) + " shot leaning, test on A task")
        # test on yesterday episode -> totally forget
        score_A = model_whole.evaluate(X_test_people, Y_test_people, verbose=1)
        print('\nTest loss: %.3f' % score_A[0])
        print('Test accuracy: %.3f' % score_A[1])
        
        all_shots_acc_A[num_train_images].append(score_A[1])
        all_shots_acc_B[num_train_images].append(score_B[1])
        K.clear_session()
        
with open('accuracy/all_shots_acc_A_without_dream_people_to_small_mammals_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_A, f)
with open('accuracy/all_shots_acc_B_without_dream_people_to_small_mammals_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_B, f)
    
####################

# load json and create model
json_file = open('models/medium_sized_mammals_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_whole = model_from_json(loaded_model_json)
# load weights into new model
model_whole.load_weights("models/medium_sized_mammals_model.h5")
print("Loaded model from disk")
# test on yesterday episode -> totally forget

# zero shot learning
print("zero shot learning")
model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("zero shot leaning, test on B task")
score = model_whole.evaluate(X_test_carnivores, Y_test_carnivores, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

print("zero shot leaning, test on A task")
# test on yesterday episode -> totally forget
score = model_whole.evaluate(X_test_medium_sized_mammals, Y_test_medium_sized_mammals, verbose=1)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])

    
all_shots_acc_A = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
all_shots_acc_B = {1:[], 
                   5:[], 
                   10:[], 
                   15:[], 
                   20:[], 
                   100:[],
                   200:[], 
                   300:[], 
                   400:[], 
                   500:[],
                   1000:[]}
for i in range(20):
    for num_train_images in nums_train_images:
        # adjust training epoch
        if num_train_images < 20:
            nb_epoch = 6
        else:
            nb_epoch = 50

        # first initialize the model and let in train on the day time task (task A)
        print(str(i))
        print(str(num_train_images) + " shot leaning, day time")
        K.clear_session()
        # load json and create model
        json_file = open('models/medium_sized_mammals_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_whole = model_from_json(loaded_model_json)
        # load weights into new model
        model_whole.load_weights("models/medium_sized_mammals_model.h5")
        #print("Loaded model from disk")

        print(str(num_train_images) + " shot leaning, night time")
        ### dreaming
        # dream(model_whole, X_train_small_mammals_var, X_train_small_mammals, Y_train_small_mammals, X_test_small_mammals, Y_test_small_mammals)
        
        print(str(num_train_images) + " shot leaning training, on second day task")
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)
        model_whole.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # network setting
        batch_size = 32
        his = model_whole.fit(X_train_carnivores[:num_train_images], Y_train_carnivores[:num_train_images], \
                  batch_size=batch_size, \
                  nb_epoch=nb_epoch, \
                  verbose=0, \
                  shuffle=True)

        print(str(num_train_images) + " shot leaning, test on B task")
        score_B = model_whole.evaluate(X_test_carnivores, Y_test_carnivores, verbose=1)
        print('\nTest loss: %.3f' % score_B[0])
        print('Test accuracy: %.3f' % score_B[1])

        print(str(num_train_images) + " shot leaning, test on A task")
        # test on yesterday episode -> totally forget
        score_A = model_whole.evaluate(X_test_medium_sized_mammals, Y_test_medium_sized_mammals, verbose=1)
        print('\nTest loss: %.3f' % score_A[0])
        print('Test accuracy: %.3f' % score_A[1])
        
        all_shots_acc_A[num_train_images].append(score_A[1])
        all_shots_acc_B[num_train_images].append(score_B[1])
        K.clear_session()
        
with open('accuracy/all_shots_acc_A_without_dream_medium_sized_mammals_to_aquatic_mammals_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_A, f)
with open('accuracy/all_shots_acc_B_without_dream_medium_sized_mammals_to_aquatic_mammals_20.txt', 'wb') as f:
    pickle.dump(all_shots_acc_B, f)
    


    