import numpy as np
import re
import sys

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def latin_to_ascii(text):
  # this is currently mapping all chars to 0xc3.
  xlate = {0xc0: 'A', 0xc1: 'A', 0xc2: 'A', 0xc3:'A', 0xc4:'A', 0xc5:'A', 0xc6:'Ae', 0xc7:'C',
    0xc8:'E', 0xc9:'E', 0xca:'E', 0xcb:'E',
    0xcc:'I', 0xcd:'I', 0xce:'I', 0xcf:'I',
    0xd0:'Th', 0xd1:'N',
    0xd2:'O', 0xd3:'O', 0xd4:'O', 0xd5:'O', 0xd6:'O', 0xd8:'O',
    0xd9:'U', 0xda:'U', 0xdb:'U', 0xdc:'U',
    0xdd:'Y', 0xde:'th', 0xdf:'ss',
    0xe0:'a', 0xe1:'a', 0xe2:'a', 0xe3:'a', 0xe4:'a', 0xe5:'a',
    0xe6:'ae', 0xe7:'c',
    0xe8:'e', 0xe9:'e', 0xea:'e', 0xeb:'e',
    0xec:'i', 0xed:'i', 0xee:'i', 0xef:'i',
    0xf0:'th', 0xf1:'n',
    0xf2:'o', 0xf3:'o', 0xf4:'o', 0xf5:'o', 0xf6:'o', 0xf8:'o',
    0xf9:'u', 0xfa:'u', 0xfb:'u', 0xfc:'u',
    0xfd:'y', 0xfe:'th', 0xff:'y'
    }
  r = ''
  for i in text:
    if ord(i) in xlate:
      r += xlate[ord(i)]
    elif ord(i) >= 0x80:
      pass
    else:
      r += str(i)
  return r

def load_pfm(fname):
  color = None
  width = None
  height = None
  scale = None
  endian = None
  file=open(fname,'rb')
  header = latin_to_ascii(file.readline().decode('utf-8')).rstrip()
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', latin_to_ascii(file.readline().decode('utf-8')))
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(latin_to_ascii(file.readline().decode('utf-8')).rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape), scale

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)


