def rotate_image(image, rotation_angle):
    return image.rotate(rotation_angle)


def image_to_greyscale(image):
    return image.convert('L')


def image_to_binary(image):
    return image.convert('1')
