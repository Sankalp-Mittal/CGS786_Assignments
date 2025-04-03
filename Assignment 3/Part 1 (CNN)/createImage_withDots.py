import os
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from PIL import Image, ImageDraw
import pickle
import imageio.v2 as imageio
from sklearn.model_selection import train_test_split

def save_compressed_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {filename}")


# Function to load images and labels
def load_data(directory):
    data_x = []
    data_y = []
    
    labels = sorted(os.listdir(directory))  # Ensure label order is consistent
    label_map = {label: idx for idx, label in enumerate(labels)}  # Mapping label names to indices
    
    for label in labels:
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):  # Ensure it's a directory
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    img = imageio.imread(img_path)  # Read image
                    data_x.append(img.tolist())  # Convert to list and append
                    data_y.append(label_map[label])  # Append label as integer
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return (data_x, data_y)

def mnist_to_dots_inverted(img_array, spacing, size, dot_radius, noise):
    """
    Converts an MNIST digit into a dot-based representation while inverting colors.
    
    Parameters:
    - img_array: The original MNIST image array.
    - spacing: Controls the density of dots.
    - size: The output image size (width, height).
    - dot_radius: Base size of the dots.
    - noise: Adds random position jitter.

    Returns:
    - PIL Image with an inverted dot representation of the digit.
    """
    img = Image.new("L", size, 0)  # Create black background image
    draw = ImageDraw.Draw(img)

    # Resize and invert MNIST image (white -> black, black -> white)
    img_array = Image.fromarray(img_array).resize(size)
    img_array = np.array(img_array)
    img_array = 255 - img_array  # Invert colors

    # Get pixel positions where intensity is low (black regions of MNIST digit)
    dot_positions = np.argwhere(img_array < 128)  

    # Uniformly sample points from the black regions
    sampled_positions = dot_positions[::spacing]

    # Convert sampled pixel positions to dots
    for y, x in sampled_positions:  
        jitter_x = np.random.randint(-dot_radius, dot_radius) if noise else 0
        jitter_y = np.random.randint(-dot_radius, dot_radius) if noise else 0
        dot_r = dot_radius + np.random.randint(-2, 2)  # Slight variation in dot size

        draw.ellipse(
            (x + jitter_x - dot_r, y + jitter_y - dot_r, x + jitter_x + dot_r, y + jitter_y + dot_r),
            fill=255,  # White dot
            outline=255
        )

    return img

def main():
    """
    Main function to generate an inverted dot-based MNIST dataset.
    """
    # Parameters
    image_size = (299, 299)  # Output image size
    dot_size = 2 # Size of individual dots
    dot_spacing = 20  # Distance between dots (higher = fewer dots)
    noise_level = 0  # Random noise in dot positions
    num_samples_train = 2  # Number of images to generate
    num_samples_test = 2  # Number of images to generate
    num_samples_valid = 2  # Number of images to generate
    output_dir = "mnist_dots"
    
    os.makedirs(output_dir, exist_ok=True)
    # create train test and validation directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    valid_dir = os.path.join(output_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    for i in range(0, 10):
        # make subdirectory for each digit
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, str(i)), exist_ok=True)


    # Load MNIST dataset
    (x_train, y_train), (x_test,y_test) = mnist.load_data()

    # Split dataset by label
    split_datasets = {i: [] for i in range(10)}

    for img, label in zip(x_train, y_train):
        split_datasets[label].append(img)
    
    for img, label in zip(x_test, y_test):
        split_datasets[label].append(img)

    # Randomly sample a subset of the dataset
    x_trainset = {i: [] for i in range(10)}
    x_testset = {i: [] for i in range(10)}
    x_validset = {i: [] for i in range(10)}
    for i in range(10):
        dataset = split_datasets[i]
        data_list = list(dataset)
        x_train = np.random.choice(len(data_list), num_samples_train, replace=False)

        # Get the sampled data
        x_train = [data_list[idx] for idx in x_train]

        x_test = np.random.choice(len(data_list), num_samples_test, replace=False)

        # Get the sampled data
        x_test = [data_list[idx] for idx in x_test]

        x_valid = np.random.choice(len(data_list), num_samples_valid, replace=False)

        # Get the sampled data
        x_valid = [data_list[idx] for idx in x_valid]

        x_trainset[i] = x_train
        x_testset[i] = x_test
        x_validset[i] = x_valid


    # Generate and save inverted dot-based MNIST images
    for label in range(10):
        dataset = x_trainset[label] 
        for i in range(num_samples_train):
            img_array = dataset[i]
            img = mnist_to_dots_inverted(img_array, dot_spacing, image_size, dot_size, noise_level)
            path = os.path.join(train_dir, str(label))
            path = os.path.join(path, f"{i}.png")
            img.save(path)
    
    for label in range(10):
        dataset = x_testset[label] 
        for i in range(num_samples_train):
            img_array = dataset[i]
            img = mnist_to_dots_inverted(img_array, dot_spacing, image_size, dot_size, noise_level)
            path = os.path.join(test_dir, str(label))
            path = os.path.join(path, f"{i}.png")
            img.save(path)
    
    for label in range(10):
        dataset = x_validset[label] 
        for i in range(num_samples_train):
            img_array = dataset[i]
            img = mnist_to_dots_inverted(img_array, dot_spacing, image_size, dot_size, noise_level)
            path = os.path.join(valid_dir, str(label))
            path = os.path.join(path, f"{i}.png")
            img.save(path)
    
    # Save the dataset in compressed format
    dataset = load_data("mnist_dots/train")
    save_compressed_pickle(dataset, "mnist_dots/train.data")
    dataset = load_data("mnist_dots/test")
    save_compressed_pickle(dataset, "mnist_dots/test.data")
    dataset = load_data("mnist_dots/valid")
    save_compressed_pickle(dataset, "mnist_dots/valid.data")

    print(f"Dot-based MNIST dataset generated in '{output_dir}' folder.")

if __name__ == "__main__":
    main()
