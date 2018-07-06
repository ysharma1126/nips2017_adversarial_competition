import os
import random
from shutil import copyfile

imgs = os.listdir('./images/')

random.seed(3)
imgs_100 = random.sample(imgs, 100)

for i in imgs_100:
	copyfile('./images/'+i, './images_100/'+i)

