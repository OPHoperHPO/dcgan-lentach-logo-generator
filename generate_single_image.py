# Lentach logo generator - DCGAN Keras.
# Modified by Anodev Development.

# Imports
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


def generate_single_image(model_path, image_save_path):
    # Save 1 generated image for demonstration purposes using matplotlib.pyplot.
    random_noise_dimension = 100
    noise = np.random.normal(0, 1, (1, random_noise_dimension))

    # Load model
    generator = load_model(model_path)

    # Predict image
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

	# Create an image using matplotlib.
    plt.imshow(generated_images[0, :], cmap='spring')
    plt.axis('off')
    plt.savefig(image_save_path)
    plt.close()


if __name__ == '__main__':
	# Image generation using pre-trained model
    generate_single_image("pretrained_models/lentach_generator_2200.h5", "output_by_pretrained_model.jpg")
    # Generate image
    generate_single_image("saved_models/lentach_generator.h5", "output.jpg")
