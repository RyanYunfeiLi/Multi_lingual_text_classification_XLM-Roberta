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

def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='linear')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=9e-6), loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model
    
    
    
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 1
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
MAX_LEN = 194
MODEL = 'jplu/tf-xlm-roberta-large'


# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)


train = pd.read_csv('train.csv', sep='\t')

test = pd.read_csv('test.csv', sep='\t')


x_train = regular_encode(np.array(train['excerpt']).tolist(), tokenizer, maxlen=MAX_LEN)
x_valid = regular_encode(np.array(test['excerpt']).tolist(), tokenizer, maxlen=MAX_LEN)
x_test = regular_encode(np.array(test['excerpt']).tolist(), tokenizer, maxlen=MAX_LEN)

y_train = np.array(train['target'])
y_valid = np.array(test['target'])


print("training shape:  ", x_train.shape)
print("validation shape:  ", x_valid.shape)
print("test shape:  ", x_test.shape)

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(9999)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)

with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()


n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


model_folder = "tf_model.h5"
os.makedirs(model_folder)

# Save the weights
model.save_weights(model_folder)

test['result'] = model.predict(valid_dataset, verbose=1)
test[['result']].to_csv('result.csv', index=False)
