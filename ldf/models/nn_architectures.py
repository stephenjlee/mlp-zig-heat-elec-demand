import sys, os
from tensorflow.keras.layers import \
    Dense, \
    Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def get_regularizer(reg_mode, reg_val):
    if reg_mode == 'L2':
        return regularizers.l2(reg_val)
    elif reg_mode == 'L1':
        return regularizers.l1(reg_val)

def define_xlarge_unrestricted_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val):
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(25, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(30, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(25, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(output_dim,
                    kernel_initializer='normal',
                    activation='linear',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    return model

