# system 
from typing import Tuple, Sequence 
from enum import Enum
import os

# lib
import numpy as np 
from dataclasses import dataclass
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import (
    Conv2D,
    Input, 
    Dense, 
    MaxPooling2D,
    Flatten
)
from keras.optimizers import Adam
from keras.models import (
    Model 
)

@dataclass
class Data:
    """
    a data class to hold data for the following
    model.

    ******** Parameters

    :parameter x_train: the x_train 
    :parameter y_train: the y_train. Should be a list 
                        of numpy arrays that are the length
                        of the number of outpus. 
    :parameter x_validate: the x_validate 
    :parameter y_train: the y_validate 
    """
    x_train: Sequence
    y_train: Sequence
    x_validate: Sequence
    y_validate: Sequence

class PretrainedWeightOptions(Enum):

    NONE = 0

@dataclass
class PModel:

    num_ouputs: int = 3
    ouput_losses: Tuple[str] = ('binary_crossentropy', 
                                'mse', 
                                'sparse_categorical_crossentropy')

    image_height: int = 224
    image_width: int = 224
    image_channels: int = 1  # grayscale 

    num_age_buckets: int = 4

    def __post_init__(self):
        self.initialize_model()

    def initialize_model(self,
                         pretrained_weights: PretrainedWeightOptions = PretrainedWeightOptions.NONE):
        """
        return a compiled keras model to do three way classification.
        This structure is based of vgg19. Thus, the provided weights 
        path must be vgg19 valid. 
        """

        img_input = Input(shape=(self.image_height, self.image_width, self.image_channels))

        # Block 1 
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Classification block
        flat = Flatten(name='flatten')(x)
        d_1 = Dense(1024, activation='relu', name='fc1')(flat)
        d_2 = Dense(1024, activation='relu', name='fc2')(flat)
        d_3 = Dense(1024, activation='relu', name='fc3')(flat)

        dem_or_rep = Dense(1, activation='sigmoid')(d_1)
        trump_perc = Dense(1, activation='sigmoid')(d_2)
        age = Dense(self.num_age_buckets, activation='softmax')(d_3)

        model = Model(inputs=[img_input], outputs=[dem_or_rep, trump_perc, age])

        model.compile(optimizer=Adam(), loss=[*self.ouput_losses])

        self.model = model 

        return model 


    def __call__(self,
                 data,
                 epochs=100, 
                 batch_size=32):

        history = self.model.fit(
            x=data.x_train,
            y=data.y_train,
            verbose=1,
            epochs=epochs,
            batch_size=batch_size
        ).history 

        return history 

if __name__ == "__main__":

    _dir = './images/'

    X = []
    Y_age = []
    Y_trump = []
    Y_party = []

    age_map = {}
    trump_map = {}
    party_map = {}

    with open('./id_age.txt') as f:
        lines = f.read().splitlines()
        age_map = {line.split()[0]: line.split()[1] for line in lines}

    with open('./id_trump.txt') as f:
        lines = f.read().splitlines()
        trump_map = {line.split()[0]: line.split()[1] for line in lines}

    with open('./id_party.txt') as f:
        lines = f.read().splitlines()
        party_map = {line.split()[0]: line.split()[1] for line in lines}

    for f in os.listdir(_dir):
        if 'jpg' in f.lower() or 'png' in f.lower():

            try:
                img = load_img(os.path.join(_dir, f), grayscale=True,
                               target_size=(224,224))

                x = img_to_array(img)

                a = int(age_map[f]) - 1
                p = int(party_map[f])
                t = float(trump_map[f])/100

                X.append(x)

                Y_age.append(a)
                Y_party.append(p)
                Y_trump.append(t)

            except:
                continue 

    X = np.array(X)
    Y = [np.array(Y_party), np.array(Y_trump), np.array(Y_age)]

    valide_p = 0.1
    validate_idx = int(len(X) * valide_p)

    X_train = X[validate_idx:]
    X_validate = X[:validate_idx]

    Y_train = list(map(lambda x: x[validate_idx:], Y))
    Y_validate = list(map(lambda x: x[:validate_idx], Y))

    data = Data(X_train, Y_train, X_validate, Y_validate)

    model = PModel()
    model(data)








