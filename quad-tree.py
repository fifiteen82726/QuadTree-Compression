from PIL import Image
import numpy as np
import math

class QT():
  def __init__(self, img, t):
    self.img = img
    self.threshold = t

  def compress(self):
    new_image = self.helper(self.img)
    self.generate_image(new_image)

  def generate_image(self, image):
    img = Image.fromarray(image.astype(np.uint8))
    img.save("quantized-file.jpg")

  def helper(self, img):
    rows, cols = img.shape
    block = [None] * 4
    block[0], block[1], block[2], block[3] = self.blockshaped(img, rows/2, cols/2)
    for b in block:
      # need to check mean, error
      if len(b[0]) > 1:
        mean, error = self.compute_mean_error(b)
        if error > self.threshold:
          b = self.helper(b)
        else:
          self.set_value_as_mean_value(b, mean)

    top = np.concatenate((block[0], block[1]), axis=1)
    down = np.concatenate((block[2], block[3]), axis=1)
    ans = np.concatenate((top, down), axis=0)

    return ans

  def set_value_as_mean_value(self, block, mean):
    rows, cols = block.shape
    for i in xrange(rows):
      for j in xrange(cols):
        block[i][j] = mean

  def blockshaped(self, arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

  def compute_mean_error(self, block):
    mean = np.mean(block)
    # error = self.mae(block, mean)
    error = self.mean_absolute_error(block, mean)

    return (mean, error)

  # maximun absolute error
  def mae(self, block, mean):
    rows, cols = block.shape
    max_v = 0
    for i in xrange(rows):
      for j in xrange(cols):
        max_v = max(abs(block[i][j] - mean), max_v)
    return max_v

  # Mean Absolute Error
  def mean_absolute_error(self, block, mean):
    rows, cols = block.shape
    v = 0
    for i in xrange(rows):
      for j in xrange(cols):
        v += abs(block[i][j] - mean)

    return v / (rows * cols)

image = Image.open('img/t1-gray.jpg').convert("L")
img = np.asarray(image)
QT(img, 25).compress()
