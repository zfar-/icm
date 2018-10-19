import keras
import numpy as np


from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Add, merge, Concatenate, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras import backend as K
import numpy as np

def build_network(input_shape, output_shape):
    state = Input(shape=input_shape)
    # , dim_ordering='th'
    h = Conv2D(32, 3, 3, padding='same',strides=(2, 2))(state)
    h = ELU(alpha=1.0)(h)
    h = Conv2D(32, 3, 3, padding='same',strides=(2, 2))(h)
    h = ELU(alpha=1.0)(h)
    h = Conv2D(32, 3, 3, padding='same',strides=(2, 2), dim_ordering='th')(h)
    h = ELU(alpha=1.0)(h)
    h = Conv2D(32, 3, 3, padding='same',strides=(2, 2))(h)
    h = ELU(alpha=1.0)(h)
    h = Flatten()(h)

    value = Dense(256, activation='relu')(h)
    value = Dense(1, activation='linear', name='value')(value)

    #policy = LSTM(output_shape, activation='sigmoid', name='policy')(h)

    policy = Dense(output_shape, activation='sigmoid', name='policy')(h)

    value_network = Model(input=state, output=value)
    policy_network = Model(input=state, output=policy)

    advantage = Input(shape=(1,))
    train_network = Model(input=[state, advantage], output=[value, policy])

    return value_network, policy_network, train_network, advantage

def build_feature_map(input_shape):
    print("Feature map input shape ", input_shape , " shape ", np.shape(input_shape))


    model = Sequential()
    model.add(Conv2D(32,(3, 3),  padding='same' ,strides=(2,2),input_shape=input_shape))
    model.add(ELU(alpha=1.0))
    model.add(Conv2D(32,(3, 3), padding ='same',strides=(2,2)))
    model.add(ELU(alpha=1.0))
    model.add(Conv2D(32,(3, 3), padding='same', strides=(2,2)))
    model.add(ELU(alpha=1.0))
    model.add(Conv2D(32,(3, 3), padding='same', strides=(2,2)))
    model.add(ELU(alpha=1.0))
    model.add(Flatten(name="features"))
    model.add(Dense(288, activation="relu"))

    model.summary()
    return model

def finalLoss(layers, lmd=1.0):
    x_1, x_2 = layers
    return (-lmd * x_1 + x_2)

def lossOfActionLayers (layers):
    x_1, x_2 = layers
    return -K.sum(x_1 + K.log(x_2 + K.epsilon()),axis=-1 )

def losses (layers , beta = 0.01):
	x_1 , x_2 = layers
	return (beta * x_1 +(1.0 - beta) * x_2 )


def lossOfFeaturedVectors(layers):
    x_1, x_2 = layers
    return (0.5 * (K.sum(K.square(x_1 - x_2), axis=-1)) / 2)


def inverse_model(output_dim=6):
    """
    s_t, s_t+1 -> a_t
    """
    def func(ft0, ft1):

        #h = merge([ft0, ft1], mode='concat')
        h = Concatenate()([ft0,ft1])
        print("Inverse concate shape :: ", K.shape(h))
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='softmax')(h)
        print(" stats ft0 {} , ft1 {} ".format(ft0, ft1))
        return h
    return func

def forward_model(output_dim=288):
    """
    s_t, a_t -> s_t+1
    """
    def func(ft, at):
        #h = merge([ft, at], mode='concat', axis=-1)
        print("Forward shape ft {} , at {}".format(ft, at))
        h = Concatenate()([ft,at])
        print("forward concate shape :: ", h)
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='linear')(h)

        return h
    return func

def build_icm_model(state_shape, action_shape, lmd=1.0, beta=0.01):

    s_t0 = Input(shape=state_shape, name="state0") # 42 x 42
    s_t1 = Input(shape=state_shape, name="state1") # 42 x 42
    a_t = Input(shape=action_shape, name="action") #6

    print("State Shape {}".format(state_shape)) #(42,42)

    reshape = Reshape(target_shape=(1,) + state_shape)  ## container

    #print("state reshape ", K.shape(reshape))
    print("State shape of after 1 + :" , K.shape( (1,) + state_shape) )
    fmap = build_feature_map( (1,) +state_shape ) ## model
    print(" feature map ", fmap )
    print("Building feature map is Done {}".format(1))

    f_t0 = fmap(reshape(s_t0))  ## container
    f_t1 = fmap(reshape(s_t1))
    print("Done for feature f_1 {}  , now feature f_2 {}".format(f_t0 , f_t1))

    act_hat = inverse_model()(f_t0, f_t1) ## Inverse Model
    f_t1_hat = forward_model()(f_t0, a_t) ## forward Model

    print("forward :: {} , inverse :: {} " .format( act_hat ,f_t1_hat))

    l_i = Lambda(lossOfActionLayers , output_shape=(1,))([a_t , act_hat])


    r_in = Lambda (lossOfFeaturedVectors,  name="reward_intrinsic" , output_shape=(1,))([f_t0, f_t1])


    loss0 = Lambda( losses , output_shape=(1,))([r_in,l_i])

    rwd =Input(shape=(1,))
    loss = Lambda(finalLoss, output_shape=(1,))([rwd,loss0])

    return Model([s_t0,s_t1,a_t,rwd], loss)

def get_reward_intrinsic(model, x):
    return K.function([model.get_layer("state0").input,
                       model.get_layer("state1").input,
                       model.get_layer("action").input],
                      [model.get_layer("reward_intrinsic").output])(x)[0]

if __name__ == "__main__":
    import numpy as np
    icm = build_icm_model((42,42), (6, ))
    icm.summary()
    print(get_reward_intrinsic(icm, [np.zeros((1, 42, 42)), np.zeros((1, 42, 42)), np.zeros((1, 6))]))
    #from keras.utils.vis_utils import plot_model
    #plot_model(icm, to_file='model.png', show_shapes=True)
