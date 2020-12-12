import tensorflow as tf
print(tf.__version__)

# 1. Get some images
# download BDD100K
!curl -s "https://2x5kv9t5uf.execute-api.us-west-2.amazonaws.com/production?func=create_download_challenge_link&filename=bdd100k"%"2Fbdd100k_images.zip" -H "Accept: */*" -o uri.txt
!xargs -n 1 curl -o "bdd100k_images.zip" < uri.txt
!unzip -q bdd100k_images.zip -d bdd100k_images
!mv ./bdd100k_images/bdd100k/images/100k ./images
!rm uri.txt
!rm bdd100k_images.zip
#you have an /images folder with .png images in it

!curl -s -O https://raw.githubusercontent.com/nyikovicsmate/thesis/dev/utils/requirements.txt
!pip3 install -q -r requirements.txt

# 2. preprocess
# resize the images
# grayscale it

# INPUT: .png images in the /images folder
# OUTPUT: .png images int the /preprocessed folder, each having the same shape ( a 2D shape, (h, w))

import pathlib
import lmdb
import cv2
import pickle
from typing import List, Tuple, Callable
import numpy as np
from tqdm import tqdm
from abc import ABC
import copy
import rawpy


class Image:
    def __init__(self, path: pathlib.Path):
        self.path = path
        if self.path.suffix == ".dng":
            with rawpy.imread(str(self.path)) as raw:
                rgb = raw.postprocess()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.data = np.array(bgr, dtype=np.uint8)
        else:
            self.data = np.array(cv2.imread(str(self.path), cv2.IMREAD_COLOR), dtype=np.uint8)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self) -> Tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]

    def grayscale(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    def resize(self, size: Tuple[int, int]):
        self.data = cv2.resize(self.data, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    def augment(self, value: int) -> "Image":
        augmented_image = copy.deepcopy(self)
        for i in range(augmented_image.size[0]):
            if i % 2 == 0:
                augmented_image.data[i][0::2] = value
            else:
                augmented_image.data[i][1::2] = value
        return augmented_image


class Store(ABC):
    def __init__(self,
                 path: pathlib.Path,
                 augment: bool):
        """
        :param path: absolute path of the dataset file/directory
        """
        self.initialized = False
        self.augment = augment
        self.path = path if path.is_absolute() else path.absolute()

    def initialize(self):
        raise NotImplementedError()

    def store(self, index: int, image: Image):
        if not self.initialized:
            self.initialize()
        self._store(index, image)

    def store_augmented(self, index: int, image: Image):
        if not self.initialized:
            self.initialize()
        self._store_augmented(index, image)

    def _store(self, index: int, image: Image):
        raise NotImplementedError()

    def _store_augmented(self, index: int, image: Image):
        raise NotImplementedError()


class LMDBStore(Store):
    def __init__(self, path: pathlib.Path, augment: bool, dataset_shape: Tuple):
        super().__init__(path, augment)
        self.dataset_shape = dataset_shape
        self.map_size = self.calculate_map_size()

    def calculate_map_size(self) -> int:
        # storing each image as unsigned 8 bit integer, so
        # size of an image = width * height * channels * 8
        # size of the whole dataset = number of images * image size
        map_size = np.prod(self.dataset_shape) * 8
        if self.augment:
            map_size *= 2
        return map_size

    def initialize(self):
        if self.path.exists():
            raise FileExistsError(f"Directory {self.path} already exists.")
        self.initialized = True

    def _store(self, index: int, image: Image):
        self._store_db(index, image, "images")

    def _store_augmented(self, index: int, image: Image):
        self._store_db(index, image, "augmented_images")

    def _store_db(self, index: int, image: Image, db_name: str):
        env = lmdb.open(str(self.path), map_size=self.map_size, max_dbs=2, readahead=False)
        db = env.open_db(key=f"{db_name}".encode("utf8"), create=True)
        with env.begin(db=db, write=True) as txn:
            txn.put(key=f"{index}".encode("utf8"), value=pickle.dumps(image.data))
        env.close()


class Processor:
    def __init__(self,
                 grayscale: bool,
                 method: Callable[[Image, Tuple[int, int]], Image],
                 size: Tuple[int, int],
                 dst: pathlib.Path,
                 src: List[pathlib.Path]):
        self.supported_extensions: List[str] = ["jpg", "png", "dng"]
        self.grayscale = grayscale
        self.method = method
        self.size = size
        self.dst = dst
        self.src = src

    def process(self):
        # print(f"Looking for images under: {chr(10)}"  # chr(10) = newline 
        #       f"{chr(10).join([str(path) for path in self.src])}")
        # print(self.src)
        paths = self._get_image_paths()
        paths = paths[:1100]
        image_cnt = len(paths)
        print(f"Found {image_cnt} images.")

        store = None
        dataset_shape = (image_cnt, *self.size) if self.grayscale else (image_cnt, *self.size, 3)
        store = LMDBStore(self.dst, False, dataset_shape)

        print(f"Processing & saving images under:{chr(10)}{store.path}")
        with tqdm(total=image_cnt) as pbar:
            for i, path in enumerate(paths):
                image = Image(path)
                # process image
                image = self.method(image, self.size)
                if self.grayscale:
                    image.grayscale()
                store.store(i, image)
                pbar.update(1)
        print("Done.")

    def _get_image_paths(self) -> List[pathlib.Path]:
        """
        Returns a list of absolute image paths found recursively starting from the scripts directory.
        """
        image_paths = []
        for ext in self.supported_extensions:
            for s in self.src:
                image_paths.extend(list(s.rglob(f"*.{ext}")))
        return image_paths

    @staticmethod
    def clip(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._clip(image, size, False)

    @staticmethod
    def clip_rnd(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._clip(image, size, True)

    @staticmethod
    def _clip(image: Image, size: Tuple[int, int], random: bool) -> Image:
        # resize image if necessary
        if image.size[0] < size[0] or image.size[1] < size[1]:
            resize_factor = np.maximum(size[0] / image.size[0], size[1] / image.size[1])
            image.resize(tuple(np.ceil(image.size * resize_factor)))
        h_idx = 0
        w_idx = 0
        if random:
            h_min = 0
            h_max = image.size[0] - size[0]
            h_idx = 0 if h_min == h_max else np.random.randint(h_min, h_max)
            w_min = 0
            w_max = image.size[1] - size[1]
            w_idx = 0 if w_min == w_max else np.random.randint(w_min, w_max)
        # clip a portion out of the image
        image.data = image.data[h_idx:h_idx+size[0], w_idx:w_idx+size[1], :]
        return image

    @staticmethod
    def scale(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._scale(image, size, False)

    @staticmethod
    def scale_rnd(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._scale(image, size, True)

    @staticmethod
    def _scale(image: Image, size: Tuple[int, int], random: bool) -> Image:
        if image.size[0] < image.size[1]:
            h = image.size[0]
            w = np.ceil(image.size[0] / (size[0] / size[1])).astype(np.int16)
        else:
            h = np.ceil(image.size[1] * (size[0] / size[1])).astype(np.int16)
            w = image.size[1]
        if image.size[0] < h or image.size[1] < w:
            resize_factor = np.maximum(h / image.size[0], w / image.size[1])
            h = np.floor(h / resize_factor).astype(np.int16)
            w = np.floor(w / resize_factor).astype(np.int16)
        h_idx = 0
        w_idx = 0
        if random:
            h_min = 0
            h_max = image.size[0] - h
            h_idx = 0 if h_min == h_max else np.random.randint(h_min, h_max)
            w_min = 0
            w_max = image.size[1] - w
            w_idx = 0 if w_min == w_max else np.random.randint(w_min, w_max)
        image.data = image.data[h_idx:h_idx+h, w_idx:w_idx+w, :]
        image.resize(size)
        return image

p = Processor(True, Processor.scale, (70, 70), pathlib.Path("./dataset").absolute(), [pathlib.Path("./images/").absolute()])
p.process()

# 3. load them into an lmdb
# https://lmdb.readthedocs.io/en/release/#named-databases
# https://realpython.com/storing-images-in-python/

# OUTPUT: images.mdb file, within the file the images are stored as 8-bit integer values in the "images" named DB
from google.colab.patches import cv2_imshow

env = lmdb.open("./dataset", readonly=True, max_dbs=2, readahead=False) 
db = env.open_db(key="images".encode("utf8"))
with env.begin(db=db) as txn:
    data = txn.get(f"0".encode("utf8"))
    image = pickle.loads(data)
env.close()

print(image.shape)

cv2_imshow(image)

# 4. get a model (a3c)

# OUTPUT: a Network class with a train(input_images: List, epochs: int, learning_rate: float) method
# input_images - list (type np.ndarray) of images NHWC format, values are in range [0,1]
# epochs - the number of epochs to train
# learning rate - the learning rate value   


from typing import Tuple, Optional, Callable
from abc import ABC, abstractmethod
from itertools import zip_longest
import copy
import cv2
import numpy as np
import tensorflow as tf
import sys
import logging
import pathlib
import time

###########
# LOGGING #
###########
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
# https://docs.python.org/3/library/logging.html#logrecord-attributes
_formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)s %(funcName)s(): %(message)s")
_handlers = [
    logging.StreamHandler(sys.stdout)   # stdout
]
for h in _handlers:
    h.setFormatter(_formatter)
    LOGGER.addHandler(h)


#########
# MODEL #
#########
class Model(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        input_shape = (None, None, 1)
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                            filters=64,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="relu",
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            bias_initializer=tf.keras.initializers.Zeros())
        self.diconv2 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=2,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.diconv3 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,2w
                                              dilation_rate=3,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.diconv4 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=4,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.actor_diconv5 = tf.keras.layers.Conv2D(filters=64,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding="same",
                                                    data_format="channels_last",
                                                    use_bias=True,
                                                    dilation_rate=3,
                                                    activation="relu",
                                                    kernel_initializer=tf.keras.initializers.he_uniform(),
                                                    bias_initializer=tf.keras.initializers.Zeros())
        self.actor_diconv6 = tf.keras.layers.Conv2D(filters=64,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding="same",
                                                    data_format="channels_last",
                                                    use_bias=True,
                                                    dilation_rate=2,
                                                    activation="relu",
                                                    kernel_initializer=tf.keras.initializers.he_uniform(),
                                                    bias_initializer=tf.keras.initializers.Zeros())
        self.actor_conv7 = tf.keras.layers.Conv2D(filters=9,  # number of actions
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding="same",
                                                  data_format="channels_last",
                                                  use_bias=True,
                                                  dilation_rate=1,
                                                  activation="softmax",
                                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                                  bias_initializer=tf.keras.initializers.Zeros())
        self.critic_diconv5 = tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding="same",
                                                     data_format="channels_last",
                                                     use_bias=True,
                                                     dilation_rate=3,
                                                     activation="relu",
                                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                                     bias_initializer=tf.keras.initializers.Zeros())
        self.critic_diconv6 = tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding="same",
                                                     data_format="channels_last",
                                                     use_bias=True,
                                                     dilation_rate=2,
                                                     activation="relu",
                                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                                     bias_initializer=tf.keras.initializers.Zeros())
        self.critic_conv7 = tf.keras.layers.Conv2D(filters=1,
                                                   kernel_size=3,
                                                   strides=1,
                                                   padding="same",
                                                   data_format="channels_last",
                                                   use_bias=True,
                                                   dilation_rate=1,
                                                   activation="linear",
                                                   kernel_initializer=tf.keras.initializers.he_uniform(),
                                                   bias_initializer=tf.keras.initializers.Zeros())

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        actor = self.actor_diconv5(x)
        actor = self.actor_diconv6(actor)
        actor = self.actor_conv7(actor)  # output shape (batch_size, height, width, 9)
        critic = self.critic_diconv5(x)
        critic = self.critic_diconv6(critic)
        critic = self.critic_conv7(critic)  # output shape (batch_size, height, width, 1)
        return actor, critic

####################
# HELPER FUNCTIONS #
####################
class Action(ABC):
    def __init__(self, img: np.ndarray):
        self._img = np.array(img, dtype=np.float32)
        self._processed_img = None

    @property
    def processed_img(self) -> np.ndarray:
        if self._processed_img is None:
            self._processed_img = np.reshape(self._process(), self._img.shape)
        return self._processed_img

    @abstractmethod
    def _process(self) -> np.ndarray:
        pass


class AlterByValue(Action):
    def __init__(self, img, value):
        super().__init__(img)
        self._value = value

    def _process(self) -> np.ndarray:
        return np.array(self._img + self._value, dtype=np.float32)


class GaussianBlur(Action):
    def __init__(self, img: np.ndarray, ksize: Tuple[int, int], sigmaX: float):
        super().__init__(img)
        self._ksize = ksize
        self._sigmaX = sigmaX

    def _process(self) -> np.ndarray:
        return cv2.GaussianBlur(self._img, ksize=self._ksize, sigmaX=self._sigmaX)


class BilaterFilter(Action):
    def __init__(self, img: np.ndarray, d: int, sigma_color: float, sigma_space: float):
        super().__init__(img)
        self._d = d
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space

    def _process(self) -> np.ndarray:
        return cv2.bilateralFilter(self._img, d=self._d, sigmaColor=self._sigma_color,
                                                  sigmaSpace=self._sigma_space)


class BoxFilter(Action):
    def __init__(self, img: np.ndarray, ddepth: int, ksize: Tuple[int, int]):
        super().__init__(img)
        self._ddepth = ddepth
        self._ksize = ksize

    def _process(self) -> np.ndarray:
        return cv2.boxFilter(self._img, ddepth=self._ddepth, ksize=self._ksize)


class MedianBlur(Action):
    def __init__(self, img: np.ndarray, ksize: int):
        super().__init__(img)
        self._ksize = ksize

    def _process(self) -> np.ndarray:
        return cv2.medianBlur(self._img, ksize=self._ksize)


class State:

    @staticmethod
    def update(img_batch: np.ndarray,
               action_batch: np.ndarray) -> np.ndarray:
        """
        :param img_batch: the bach of input images in NHWC format
        :param action_batch: action img.shape-d mx with values 0-(N_ACTIONS-1)
        :return: the modified image batch
        """
        assert img_batch.shape == action_batch.shape
        img_batch_new = np.zeros_like(img_batch)
        for n in range(img_batch.shape[0]):
            img = img_batch[n]
            actions = action_batch[n]
            img_alternatives = {
                0: AlterByValue(img, -1.0 / 255.0),
                1: AlterByValue(img, 0.0),
                2: AlterByValue(img, 1.0 / 255.0),
                3: GaussianBlur(img, ksize=(5, 5), sigmaX=0.5),
                4: BilaterFilter(img, d=5, sigma_color=0.1, sigma_space=5),
                5: MedianBlur(img, ksize=5),
                6: GaussianBlur(img, ksize=(5, 5), sigmaX=1.5),
                7: BilaterFilter(img, d=5, sigma_color=1.0, sigma_space=5),
                8: BoxFilter(img, ddepth=-1, ksize=(5, 5))
            }
            for k, v in img_alternatives.items():
                img_batch_new[n] = np.where(actions == k, v.processed_img, img_batch_new[n])

        return img_batch_new

class Network:

    @property
    def noise_func(self):
        return self._noise_func

    @noise_func.setter
    def noise_func(self, func: Callable):
        if not isinstance(func, Callable):
            raise TypeError("Noise function must be an instance of `typing.Callable`.")
        if func.__code__.co_argcount != 1:
            raise AttributeError(
                f"Noise function must have exactly 1 parameter, the image which to process. Got {func.__code__.co_varnames}")
        test_img = np.zeros(shape=(1, 10, 10, 3), dtype=np.float32)
        ret = func(test_img)
        if type(ret) is not np.ndarray and len(ret.shape) != 4:
            raise TypeError(
                "Noise function must return 4D numpy.ndarray type with (batch, height, width, channel) dimensions.")
        self._noise_func = func

    @staticmethod
    def _normalized_noise_func(images: np.ndarray) -> np.ndarray:
        """
        Adds noise to image.
        :param images: A batch of images, 4D array (batch, height, width, channels)
        :return: The noisy batch of input images.
        """
        fill_value = 0.5
        try:
            # this will fail unless there is exactly 4 dimensions to unpack from
            batch, height, width, channels = images.shape
        except ValueError:
            raise TypeError(f"Image must be a 4D numpy array. Got shape {images.shape}")
        if channels == 1:
            for img in images:
                for h in range(height):
                    if h % 2 == 0:
                        img[h][0::2] = [fill_value]
                    else:
                        img[h][1::2] = [fill_value]
        elif channels == 3:
            for img in images:
                for h in range(height):
                    if h % 2 == 0:
                        img[h][0::2] = [fill_value, fill_value, fill_value]
                    else:
                        img[h][1::2] = [fill_value, fill_value, fill_value]
        else:
            raise ValueError(f"Unsupported number of image dimensions, got {channels}")
        return images

    def __init__(self):
        self.model = Model()
        self._steps_per_episode = 5
        self._discount_factor = 0.95
        self._noise_func = Network._normalized_noise_func

    def predict(self, x: np.array, *args, **kwargs) -> tf.Tensor:
        x = tf.convert_to_tensor(self.noise_func(x))
        s_t0_channels = tf.split(x, num_or_size_splits=x.shape[-1], axis=3)
        result = None
        for s_t0 in s_t0_channels:
            for t in range(self._steps_per_episode):
                # predict the actions and values
                a_t, _ = self.model(s_t0)
                # sample the actions
                sampled_a_t = self._sample_most_probable(a_t)
                # update the current state/image with the predicted actions
                s_t1 = tf.convert_to_tensor(State.update(s_t0.numpy(), sampled_a_t.numpy()), dtype=tf.float32)
                s_t0 = s_t1
            result = s_t0 if result == None else tf.concat([result, s_t0], axis=3)
        return result.numpy()


    def _train_step(self, x, y, optimizer):
        s_t0 = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        episode_r = 0
        r = {}  # reward
        V = {}  # expected total rewards from state
        past_action_log_prob = {}
        past_action_entropy = {}
        with tf.GradientTape() as tape:
            for t in range(self._steps_per_episode):
                # predict the actions and values
                a_t, V_t = self.model(s_t0)
                # sample the actions
                sampled_a_t = self._sample_random(a_t)
                # clip distribution into range to avoid 0 values, which cause problem with calculating logarithm
                a_t = tf.clip_by_value(a_t, 1e-6, 1)
                a_t_log = tf.math.log(a_t)
                past_action_log_prob[t] = self._mylog_prob(a_t_log, sampled_a_t)
                past_action_entropy[t] = self._myentropy(a_t, a_t_log)
                V[t] = V_t
                # update the current state/image with the predicted actions
                s_t1 = tf.convert_to_tensor(State.update(s_t0.numpy(), sampled_a_t.numpy()), dtype=tf.float32)
                r_t = self._mse(y, s_t0, s_t1)
                r[t] = tf.cast(r_t, dtype=tf.float32)
                s_t0 = s_t1
                # print(tf.reduce_mean(r_t))
                episode_r += tf.reduce_mean(r_t) * tf.math.pow(self._discount_factor, t)

            R = 0
            actor_loss = 0
            critic_loss = 0
            beta = 1e-2
            for t in reversed(range(self._steps_per_episode)):
                R *= self._discount_factor
                R += r[t]
                A = R - V[t]  # advantage
                # Accumulate gradients of policy
                log_prob = past_action_log_prob[t]
                entropy = past_action_entropy[t]

                # Log probability is increased proportionally to advantage
                actor_loss -= log_prob * A
                # Entropy is maximized
                actor_loss -= beta * entropy
                actor_loss *= 0.5  # multiply loss by 0.5 coefficient
                # Accumulate gradients of value function
                critic_loss += (R - V[t]) ** 2 / 2

            total_loss = tf.reduce_mean(actor_loss + critic_loss)
            actor_grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        return episode_r, total_loss

    def train(self, lmdb_dataset, shape, nr_of_images, epochs, learning_rate):
        gen = self.generator(lmdb_dataset, shape,nr_of_images, batch_size=50)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)
        # train
        for e_idx in range(epochs):
            train_loss = tf.constant(0, dtype=tf.float32)
            start_sec = time.time()
            # process a batch
            x_b, y_b = next(gen)
            x_b = self.noise_func(x_b)
            # print(x_b.shape)
            # print(y_b.shape)
            episode_r, train_loss = self._train_step(x_b, y_b, optimizer)
            delta_sec = time.time() - start_sec
            print(f"Epoch: {e_idx} episode_r: {episode_r:.2f} train_loss: {train_loss:.2f} train_time: {delta_sec:.2f}s")

    def generator(self, lmdb_dataset: pathlib.Path, shape: tuple, nr_of_images: int, batch_size: int) -> np.array:
        x = np.zeros((batch_size, *shape, 1), dtype=np.float32)
        y = np.zeros((batch_size, *shape, 1), dtype=np.float32)
        while True: 
            env = lmdb.open(str(lmdb_dataset), readonly=True, max_dbs=2, readahead=False) 
            db = env.open_db(key="images".encode("utf8"))
            indexes = np.random.randint(0, nr_of_images, batch_size)

            with env.begin(db=db) as txn:
                for i, idx in enumerate(indexes):
                    data = txn.get(str(idx).encode("utf8"))
                    image = pickle.loads(data)
                    image = np.array(image[:,:,np.newaxis] / 255.0, dtype=np.float32)
                    x[i] = image
                    y[i] = image
            env.close()
            yield x,y

    @staticmethod
    def _mse(a, b, c):
        """
        Calculates the mean squared error for image batches given by the formula:
        mse = (a-b)**2 - (a-c)**2
        :param a:
        :param b:
        :param c:
        :return:
        """
        mse = tf.math.square(a - b) * 255
        mse -= tf.math.square(a - c) * 255
        return mse

    @staticmethod
    def _myentropy(prob, log_prob):
        return tf.stack([- tf.math.reduce_sum(prob * log_prob, axis=3)], axis=3)

    @staticmethod
    def _mylog_prob(data, indexes):
        """
        Selects elements from a multidimensional array.
        :param data: The 4D actions vector with logarithmic values.
        :param indexes: The indexes to select.
        :return: The selected indices from data eg.: data=[[11, 2], [3, 4]], indexes=[[0],[1]] --> [[11], [4]]
        """
        data_flat = tf.reshape(data, (-1, data.shape[-1]))
        indexes_flat = tf.reshape(indexes, (-1,))
        one_hot_mask = tf.one_hot(indexes_flat, data_flat.shape[-1], on_value=True, off_value=False, dtype=tf.bool)
        output = tf.boolean_mask(data_flat, one_hot_mask)
        return tf.reshape(output, (*data.shape[0:-1], 1))

    @staticmethod
    def _sample_random(distribution):
        """
        Samples the image action distribution returned by the last softmax activation.
        :param distribution: A 4D array with probability distributions shaped (batch_size, height, width, samples)
        :return: The sampled 4D vector shaped of (batch_size, height, width, 1)
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.math.log(d)
        d = tf.random.categorical(logits=d, num_samples=1)  # draw samples from the categorical distribution
        d = tf.reshape(d, (*distribution.shape[0:-1], 1))
        return d

    @staticmethod
    def _sample_most_probable(distribution):
        """
        Samples the image action distribution returned by the last softmax activation by returning the
        most probable action indexes from samples.
        :param distribution: A 4D array with probability distributions shaped (batch_size, height, width, samples)
        :return: The sampled 4D vector shaped of (batch_size, height, width, 1)
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.argmax(d, axis=1)
        d = tf.reshape(d, (*distribution.shape[0:-1], 1))
        return d

network = Network()

# 5. train the model
# input_images: np.ndarray = np.random.randint(0, 256, (40, 10, 10, 1)) / 255.0
# input_images = input_images.astype(np.float32)
epochs = 500
learning_rate = 1e-4
network.train(pathlib.Path("./dataset").absolute(), (70, 70), 1000, epochs, learning_rate)

# 6. evaluate the performance
# using metrics PSNR, SSIM, 

from google.colab.patches import cv2_imshow

# get 10 evaluation images outside the train image range
# train idx 0-1000 , total idx. 0-1100
indexes = [i for i in range(1001, 1011)]
images = []
env = lmdb.open("./dataset", readonly=True, max_dbs=2, readahead=False) 
db = env.open_db(key="images".encode("utf8"))
with env.begin(db=db) as txn:
    for i in indexes:
        data = txn.get(str(i).encode("utf8"))
        image = pickle.loads(data)
        images.append(image[:,:,np.newaxis] / 255.0)
env.close()
# predict
predictions = network.predict(np.array(images, dtype=np.float32))

for true, pred in zip(images, predictions):
    print(f"PSNR {tf.image.psnr(true, pred, 1).numpy()}")
    print(f"SSIM {tf.image.ssim(tf.image.convert_image_dtype(true, tf.float32), tf.image.convert_image_dtype(pred, tf.float32), 1).numpy()}")
    print("")

for true, pred in zip(images, predictions):
    cv2_imshow(np.concatenate((cv2.resize(true, (200, 200)), cv2.resize(pred, (200, 200))), axis=1) * 255.0)







