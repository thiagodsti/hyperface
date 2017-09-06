import cv2
import numpy as np
import argparse
import chainer
import os

import config
import drawing
import log_initializer
import models

# logging
from logging import getLogger, DEBUG
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)

# Disable type check in chainer
os.environ["CHAINER_TYPE_CHECK"] = "0"

def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, chainer.variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v

def _draw_gender(img, gender, size=7, idx=0):
    # Upper right
    pt = (img.shape[1] - (size + 5) * (2 * idx + 1), size + 5)
    if gender == 0:
        _draw_circle(img, pt, (255, 0.3, 0.3), size, -1)  # male
    elif gender == 1:
        _draw_circle(img, pt, (0.3, 0.3, 255), size, -1)  # female

def _draw_circle(img, pt, color, radius=4, thickness=-1):
    pt = (int(pt[0]), int(pt[1]))
    cv2.circle(img, pt, radius, color, int(thickness))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='HyperFace training script')
  parser.add_argument('--config', '-c', default='config.json',
                      help='Load config from given json file')
  parser.add_argument('--model', required=True, help='Trained model path')
  parser.add_argument('--input', required=True, help='Input video path')
  parser.add_argument('--output', required=True, help='Output video path')
  args = parser.parse_args()


  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(args.input)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  logger.info('HyperFace Evaluation')

  # Load config
  config.load(args.config)

  # Define a model
  logger.info('Define a HyperFace model')
  model = models.HyperFaceModel()
  model.train = False
  model.report = False
  model.backward = False

  # Initialize model
  logger.info('Initialize a model using model "{}"'.format(args.model))
  chainer.serializers.load_npz(args.model, model)

  # Setup GPU
  if config.gpu >= 0:
      chainer.cuda.check_cuda_available()
      chainer.cuda.get_device(config.gpu).use()
      model.to_gpu()
      xp = chainer.cuda.cupy
  else:
      xp = np

  success,image = cap.read()
  count = 0
  success = True
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  print(frame_width)
  print(frame_height)

  fourcc = cv2.cv.CV_FOURCC(*'XVID')
  print(fourcc)
  out = cv2.VideoWriter(args.output,fourcc, 30.0, (frame_width,frame_height))
  while success:
    success,image = cap.read()

    print('Read a new frame: ', success)
    count += 1
    if (success):
      # Load image file
      img = image
      img2 = image
      if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
          logger.error('Failed to load')
          exit()
      img = img.astype(np.float32) / 255.0  # [0:1]
      img = cv2.resize(img, models.IMG_SIZE)
      img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
      img = np.transpose(img, (2, 0, 1))

      # Create single batch
      imgs = xp.asarray([img])
      x = chainer.Variable(imgs, volatile=True)

      # Forward
      logger.info('Forward the network')
      y = model(x)

      # Chainer.Variable -> np.ndarray
      imgs = _cvt_variable(y['img'])
      detections = _cvt_variable(y['detection'])
      landmarks = _cvt_variable(y['landmark'])
      visibilitys = _cvt_variable(y['visibility'])
      poses = _cvt_variable(y['pose'])
      genders = _cvt_variable(y['gender'])

      # Use first data in one batch
      img = imgs[0]
      detection = detections[0]
      landmark = landmarks[0]
      visibility = visibilitys[0]
      pose = poses[0]
      gender = genders[0]

      img = np.transpose(img, (1, 2, 0))
      img = img.copy()
      img += 0.5  # [-0.5:0.5] -> [0:1]
      detection = (detection > 0.5)
      gender = (gender > 0.5)

      # Draw results
      drawing.draw_detection(img, detection)
      landmark_color = (0, 255, 0) if detection == 1 else (0, 0, 255)
      drawing.draw_landmark(img, landmark, visibility, landmark_color, 0.5)
      # Descobrir as outras poses o que querem dizer em relacao a cabeca.
      # pose[0] -> cabeca virada pros lados roll (rolando)
      # pose[1] -> cabeca virada pra cima ou baixo pitch
      # pose[2] -> cabeca olhando pra qual lado yaw
      limite = 0.1
      if (pose[1]<-limite):
          print ('Cabeca em estado para Baixo')
      elif(pose[1]>limite):
          print('Cabeca em estado para Cima');
      elif(pose[1]>=-limite and pose[2]<=limite):
          print('Cabeca em estado Reto');
      if (pose[2]<-limite):
          print ('Olhando para a esquerda')
      elif (pose[2]>limite):
          print('Olhando para a direita')
      elif (pose[2]>=-limite and pose[2]<=limite):
          print('Olhando para frente')
      print(pose);
      drawing.draw_pose(img, pose)
      drawing.draw_gender(img, gender)

      drawing.draw_detection(img2, detection)
      drawing.draw_landmark(img2, landmark, visibility, landmark_color, 0.5)
      drawing.draw_pose(img2, pose)
      drawing.draw_gender(img2, gender)
      out.write(img2)
      # Show image
      #logger.info('Show the result image')
      #cv2.imshow('result', img)
      #cv2.waitKey(0)
      cv2.imwrite("frames/frame%d.jpg" % count, img2)     # save frame as JPEG file

  # When everything done, release the video capture object
  cap.release()
  out.release()

  # Closes all the frames
  cv2.destroyAllWindows()
