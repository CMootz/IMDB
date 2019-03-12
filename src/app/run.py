from app import Preprocessing
from app import Train
from app import Predict
from matplotlib import pyplot as plt
import os

#numbers = Preprocessing('numbers')
#numbers.load_data('zip.csv', name='raw', sep=' ', header=None)
#numbers.cleanup('raw', drop=257)
#numbers.split_data(target=0)
#train = Train(numbers, epoch=100000)

image_dir = os.path.dirname(__file__)
image_path = image_dir + '../../data/numbers/own_digits/'
picturelist = os.listdir(image_path)
for item in picturelist:
    predict = Predict(image_path+item)
    os.system(f'say {predict.y_pred.item()}')
#plt.plot(train.loss_hist_train)
#plt.plot(train.loss_hist_val)
#plt.show()
