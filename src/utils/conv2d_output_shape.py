
import argparse

def conv_output_shape(h_w, kernel_size = 1, stride = 1, pad = 0, dilation = 1):
  """
  Utility function for computing output of convolutions
  takes a tuple of (h,w) and returns a tuple of (h,w)
  """

  if type(h_w) is not tuple:
      h_w = (h_w, h_w)

  if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)

  if type(stride) is not tuple:
      stride = (stride, stride)

  if type(pad) is not tuple:
      pad = (pad, pad)

  h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
  w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1

  return h, w

def main():
  parser = argparse.ArgumentParser(description='Conv2d output shape calculator')
  parser.add_argument('--height', type = int)
  parser.add_argument('--width', type = int)
  parser.add_argument('--kernel_size', type = int)
  parser.add_argument('--stride', type = int)
  parser.add_argument('--padding', type = int, default = 0)
  parser.add_argument('--dilatation', type = int, default = 1)
  args = parser.parse_args()
  
  new_h, new_w = conv_output_shape((args.height, args.width), args.kernel_size, args.stride, args.padding, args.dilatation)
  
  print("{}x{}".format(new_h, new_w))
  
if __name__ == '__main__':
  main()
