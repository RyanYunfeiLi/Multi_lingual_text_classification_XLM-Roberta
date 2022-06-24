import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow import keras


num_classes = 7
AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 3
# BATCH_SIZE = 4 * strategy.num_replicas_in_sync
BATCH_SIZE = 32
MAX_LEN = 128
MODEL = 'jplu/tf-xlm-roberta-large'

def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = tf.keras.layers.Flatten()(sequence_output)
    out = Dense(num_classes, activation='softmax')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=9e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        #return_attention_masks=False, 
        return_token_type_ids=False,
        padding='max_length',
        pad_to_max_length=True,
        max_length=maxlen
    )
    res = [x[:maxlen] for x in enc_di['input_ids']]
    return np.asarray(res)


tokenizer = AutoTokenizer.from_pretrained(MODEL)
# only load model weight, keey model layer consistent as before
transformer_layer = TFAutoModel.from_pretrained(MODEL)
model = build_model(transformer_layer, max_len=MAX_LEN)


checkpoint_path = "tf_model5.h5/tf_model5.h5"


model.load_weights(checkpoint_path) 

model.summary()

train = pd.read_csv('train.tsv', sep='\t')

test = pd.read_csv('dev.tsv', sep='\t')

valid = pd.read_csv('test/test.tsv', sep='\t')


data_set = pd.concat([train, test, valid])


x_valid = regular_encode(np.array(valid['query']).tolist(), tokenizer, maxlen=MAX_LEN)

y_valid = tf.keras.utils.to_categorical(np.asarray(data_set['domain'].factorize()[0][train.shape[0] + test.shape[0]:]), num_classes=num_classes)


valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

# evaluate the model
loss, acc = model.evaluate(x_valid, y_valid, verbose=2)
print("Loss: ", loss)
print("Accuracy: ", acc)
