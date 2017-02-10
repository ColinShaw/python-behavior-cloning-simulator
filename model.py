from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda
import numpy as np
import cv2
import os

def load_image(filename):
    filename = os.getcwd() + '/training-data/' + filename.decode('utf-8').strip()
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[70:105,:,:]
    return cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)

def load_data():
    images = []
    angles = []
    csv_file = os.getcwd() + '/training-data/driving_log.csv'
    csv_data = np.genfromtxt(csv_file, delimiter=',', dtype=None)
    for row in csv_data:
        images.append(load_image(row[0]))
        angles.append(row[3])
        images.append(load_image(row[1]))
        angles.append(row[3] + 0.2)
        images.append(load_image(row[2]))
        angles.append(row[3] - 0.2) 
    return {'images': np.array(images), 'angles': np.array(angles)}

def adjust_luminance(image, angle):
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
    image[:,:,1] = image[:,:,1] * (0.25 + np.random.uniform(0.25, 0.75))
    return cv2.cvtColor(image, cv2.COLOR_HLS2RGB), angle

def horizontal_translate(image, angle):
    x = 20 * np.random.uniform() - 10
    angle = angle + x / 200
    image = cv2.warpAffine(image, np.float32([[1,0,x],[0,1,0]]), (64,64))
    return image, angle

def horizontal_flip(image, angle):
    if np.random.uniform(0,100) > 50:
        return cv2.flip(image, 1), -angle	
    return image, angle

def augmentation_pipeline(image, angle):
    image, angle = adjust_luminance(image, angle)
    image, angle = horizontal_translate(image, angle)
    image, angle = horizontal_flip(image, angle)
    return image, angle

def pipeline_generator(data, batch_size):
    while True:
        images = []
        angles = []
        while len(images) < batch_size:
            index = np.random.randint(len(data))
            image = data['images'][index] 
            angle = data['angles'][index]
            image, angle = augmentation_pipeline(image, angle)
            images.append(image)
            angles.append(angle)
        yield np.array(images), np.array(angles)

def save_model(model, filename):
    file = open(filename, 'w')
    file.write(model.to_json())
    file.close()

model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3)),
    Convolution2D(3, 1, 1, init='glorot_normal'),
    Convolution2D(16, 5, 5, activation='elu', init='glorot_normal'),
    Convolution2D(16, 5, 5, activation='elu', init='glorot_normal'),
    MaxPooling2D((2,2), strides=(2,2)),
    Dropout(0.5),
    Convolution2D(32, 5, 5, activation='elu', init='glorot_normal'),
    Convolution2D(32, 5, 5, activation='elu', init='glorot_normal'),
    MaxPooling2D((2,2), strides=(2,2)),
    Dropout(0.5),
    Convolution2D(64, 3, 3, activation='elu', init='glorot_normal'),
    Convolution2D(64, 3, 3, activation='elu', init='glorot_normal'),
    MaxPooling2D((2,2), strides=(2,2)),
    Dropout(0.5),
    Flatten(),
    Dense(1024, activation='elu', init='glorot_normal'),
    Dropout(0.3),
    Dense(128, activation='elu', init='glorot_normal'),
    Dropout(0.2),
    Dense(16, activation='elu', init='glorot_normal'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit_generator(pipeline_generator(load_data(),1000), samples_per_epoch=20000, nb_epoch=10)
model.save_weights('model.h5')
save_model(model, 'model.json')
