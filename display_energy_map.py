import os
import numpy as np
import cv2


# define a function to get the energy map
def get_energy_map(img):
    # convert to a grayscale image if it's a color image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # make the datatype as float64 for calculation
    img = img.astype(np.float64)
    # use the sobel filter for edge detection in both x and y directions, add some border
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_CONSTANT)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_CONSTANT)

    # add the absolute values of gradient_x and gradient_y
    energy_map = np.abs(gradient_x) + np.abs(gradient_y)

    return energy_map


def visualize_energy_map(energy_map):
    # normalize the energy map for display purpose
    energy_map_vis = (energy_map / np.max(energy_map) * 255).astype(np.uint8)
    return energy_map_vis


def main():
    # read in image
    source_dir = '.\src'
    dolphin = os.path.join(source_dir, 'dolphin.png')
    img = cv2.imread(dolphin, 1)

    # calculate the energy map
    energy_map = get_energy_map(img)
    energy_map_vis = visualize_energy_map(energy_map)

    # display the image
    cv2.imshow('image', img.astype(np.uint8))
    cv2.imshow('energy_map', energy_map_vis)

    # if push 'Esc', destroy the display
    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyAllWindows()

    # if push 's', save the image
    elif k == ord('s'):

        # define the location to save the energy map
        out_dir = '.\dst'

        # make the directory if not exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_energy_map = os.path.join(out_dir, 'dolphin_energy_map.png')
        # write the energy map and destroy the window
        cv2.imwrite(out_energy_map, energy_map_vis)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()