import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import string
import numpy as np
import pandas as pd
from numpy import array
from pickle import load
from PIL import Image
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import sys, time, warnings
warnings.filterwarnings("ignore")
import re
import keras
import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

app = Flask(__name__, static_folder = "gallery")

#query_string=""

#PREPROCESSING THE DATA
dir_Flickr_text = "Flickr8k_text/Flickr8k.token.txt"
train_images_path = 'Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_path = 'Flickr8k_text/Flickr_8k.testImages.txt'
image_path = 'Flicker8k_Dataset/'
jpgs = os.listdir(image_path)

print("Total Images in Dataset = {}".format(len(jpgs)))


# In[73]:


file = open(dir_Flickr_text,'r')
text = file.read()
file.close()

datatxt = []
for line in text.split('\n'):
    col = line.split('\t')
    if len(col) == 1:
        continue
    w = col[0].split("#")
    datatxt.append(w + [col[1].lower()])

data = pd.DataFrame(datatxt,columns=["filename","index","caption"])
data = data.reindex(columns =['index','filename','caption'])
data = data[data.filename != '2258277193_586949ec62.jpg.1']
uni_filenames = np.unique(data.filename.values)


# In[74]:



vocabulary = []
for txt in data.caption.values:
   vocabulary.extend(txt.split())


# In[6]:


def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(string.punctuation)
    return(text_no_punctuation)

def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return(text_len_more_than1)

def remove_numeric(text):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if isalpha:
            text_no_numeric += " " + word
    return(text_no_numeric)

def text_clean(text_original):
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    return(text)

for i, caption in enumerate(data.caption.values):
    newcaption = text_clean(caption)
    data["caption"].iloc[i] = newcaption


# In[75]:


clean_vocabulary = []   #LIST OF WORDS IN DATASET
for txt in data.caption.values:
    clean_vocabulary.extend(txt.split())


# In[76]:


PATH = "Flicker8k_Dataset/"
all_captions = []
for caption  in data["caption"].astype(str):
    caption = '<start> ' + caption+ ' <end>'
    all_captions.append(caption)


# In[77]:

#PATH TO ALL IMAGES
all_img_name_vector = []
for annot in data["filename"]:
    full_image_path = PATH + annot
    all_img_name_vector.append(full_image_path)


# In[10]:

#LIMITING NO. OF CAPTIONS TO 40,000
def data_limiter(num,total_captions,all_img_name_vector):
    train_captions, img_name_vector = shuffle(all_captions,all_img_name_vector,random_state=1)
    train_captions = train_captions[:num]
    img_name_vector = img_name_vector[:num]
    return train_captions,img_name_vector

train_captions,img_name_vector = data_limiter(40000,all_captions,all_img_name_vector)


# In[78]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img, image_path

image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output # SOFTMAX FN
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# In[12]:


encode_train = sorted(set(img_name_vector))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)


# In[13]:

#REPLACE UNKNOWN WORDS WITH <UNK>
top_k = 8000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                 oov_token="<unk>",
                                                 filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[79]:


def calc_max_length(tensor):
    return max(len(t) for t in tensor)
max_length = calc_max_length(train_seqs)

def calc_min_length(tensor):
    return min(len(t) for t in tensor)
min_length = calc_min_length(train_seqs)


# In[80]:

#SPLITTING THE DATASET
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,cap_vector, test_size=0.2, random_state=0)


# In[81]:


BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
features_shape = 512
attention_features_shape = 49


# In[82]:

#LOADING THE EXTRACTED IMAGE FEATURES
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
#dataset contains extracted image features
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
         num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[83]:

print('defining models')
class VGG16_Encoder(tf.keras.Model):
    # This encoder passes the features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(VGG16_Encoder, self).__init__()
        # shape after fc == (batch_size, 49, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)  #256 NEURONS
        
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x 


# In[21]:


'''The encoder output(i.e. 'features'), hidden state(initialized to 0)(i.e. 'hidden') and
the decoder input (which is the start token)(i.e. 'x') is passed to the decoder.'''

class Rnn_Local_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Rnn_Local_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #This layer can only be used as the first layer in a model.
        self.gru = tf.keras.layers.GRU(self.units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
  
        self.fc1 = tf.keras.layers.Dense(self.units)

        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
        self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        #Batch normalization applies a transformation that maintains the mean output close to 0
        #and the output standard deviation close to 1.
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        # Implementing Attention Mechanism
        self.Uattn = tf.keras.layers.Dense(units)
        self.Wattn = tf.keras.layers.Dense(units)
        self.Vattn = tf.keras.layers.Dense(1)

    def call(self, x, features, hidden):
        # features shape ==> (64,49,256) ==> Output from ENCODER
        # hidden shape == (batch_size, hidden_size) ==>(64,512)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (64, 49, 1)
   # Attention Function
        '''e(ij) = f(s(t-1),h(j))'''
        ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''

        score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))

   # self.Uattn(features) : (64,49,512)
   # self.Wattn(hidden_with_time_axis) : (64,1,512)
   # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
   # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score

   # you get 1 at the last axis because you are applying score to self.Vattn
   # Then find Probability using Softmax
        '''attention_weights(alpha(ij)) = softmax(e(ij))'''

        attention_weights = tf.nn.softmax(score, axis=1)

        # attention_weights shape == (64, 49, 1)
        # Give weights to the different pixels in the image
        ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) '''

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

   # Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
   # context_vector shape after sum == (64, 256)
   # x shape after passing through embedding == (64, 1, 256)

        x = self.embedding(x)
       # x shape after concatenation == (64, 1,  512)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
       # passing the concatenated vector to the GRU

        output, state = self.gru(x)
       # shape == (batch_size, max_length, hidden_size)

        x = self.fc1(output)
       # x shape == (batch_size * max_length, hidden_size)

        x = tf.reshape(x, (-1, x.shape[2]))

       # Adding Dropout and BatchNorm Layers
        x= self.dropout(x)
        x= self.batchnormalization(x)

       # output shape == (64 * 512)
        x = self.fc2(x)

       # shape : (64 * 8329(vocab))
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


encoder = VGG16_Encoder(embedding_dim)
decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)


# In[22]:

#Adam optimization is a stochastic gradient descent method 
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
   from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# In[23]:


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[84]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)


# In[28]:


@tf.function
def train_step(img_tensor, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image

    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss


# In[86]:

print("starting training")
EPOCHS = 30
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[87]:


def evaluate(image):
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result
'''
image_id = 'Flicker8k_Dataset/2521770311_3086ca90de.jpg'
result = evaluate(image_id)
for i in result:
    if i=="<unk>":
        result.remove(i)

#remove <end> from result        
result_join = ' '.join(result)
result_final = result_join.rsplit(' ', 1)[0]
print ('Prediction Caption:', result_final)
'''

#app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
image_names = []

@app.route("/",methods=['POST','GET'])
def temp():
    return render_template("home.html")

f = open('data.json',)
# returns JSON object as a dictionary
data = json.load(f)


@app.route("/renderupload", methods=['POST','GET'])
def showuploadpage():
    return render_template('upload.html')

@app.route("/upload", methods = ['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'gallery/')
    print(target)
    if not os.path.isdir(target):   #if folder does not exist
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target,filename])
        print(destination)
        file.save(destination)
        image_id = destination 
        #image_id = filename 
        #image_id = APP_ROOT+'gallery\\' + filename
        result = evaluate(image_id) #generates caption
        for i in result:
            if i=="<unk>":
                result.remove(i)
        result_join = ' '.join(result)
        result_final = result_join.rsplit(' ', 1)[0]
        print ('Prediction Caption:', result_final)
        predicted_caption = result_final
        #abs_id = '../gallery\\' + filename                        #changed later
        image_names.append(filename)
        image_id = filename
        tag=request.form["tag"]
        new_pic = {"id" : image_id, "caption":[predicted_caption],"tags":[tag]}
        def write_json(data, filename='data.json'):
            with open(filename,'w') as f:
                json.dump(data, f, indent=4)
        with open('data.json') as json_file:
            data = json.load(json_file)
            temp = data['pics']
            # appending data 
            temp.append(new_pic)
        write_json(data)
        #file.save(destination)
    return render_template("complete.html")


from nltk.corpus import stopwords
from nltk import download



@app.route('/rendersearch', methods = ['POST','GET'])
def home():
    return render_template('search.html',image = 'searchicon.png')


from rank_bm25 import BM25Okapi

@app.route('/search',methods=['POST','GET'])
def query():
    if request.method=='POST':
        query=request.form['query']
        query_tag=request.form['searchtag']
        
        tagged_images=list()

        for i in data['pics']:                          #to check for images with matching tag, and storing their ids in tagged_images
            for j in i['tags']:
                if query_tag == j:
                    tagged_images.append(i['id'])       #getting list of captions
        list_of_caption = list()
        if len(tagged_images)==0:                       #if there are no images with matching tag
            for i in data['pics']:
                for j in i['caption']:
                    list_of_caption.append(j)
        else:
            for i in data['pics']:
                if i['id'] in tagged_images:
                    for j in i['caption']:
                        list_of_caption.append(j)
        #print(list_of_caption)
        tok_text = list()   #tokenized text
        for i in list_of_caption:
            j = word_tokenize(i)
            tok_text.append(j)
        bm25 = BM25Okapi(tok_text)
        tokenized_query = query.lower().split(" ")
        results = bm25.get_top_n(tokenized_query, tok_text, n=3)
        print(results)
        indexes=list()
        for i in results:
            if i in tok_text:
                indexes.append(tok_text.index(i))
        
        list_of_ids=list()
        for i in data["pics"]:
            list_of_ids.append(i["id"])
        
        retreived_images=list()

        for i in indexes:
            retreived_images.append(list_of_ids[i])
        print(retreived_images)
        return render_template('result_new.html',image1=retreived_images[0],image2=retreived_images[1],image3=retreived_images[2],query=query)


if __name__ == '__main__':
    app.run(debug=True)