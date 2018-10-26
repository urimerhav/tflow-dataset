import tensorflow as tf
from .pair_generator import PairGenerator
from .model import Inputs


class Dataset(object):
    img1_resized = 'img1_resized'
    img2_resized = 'img2_resized'
    label = 'same_person'

    def __init__(self, generator=PairGenerator()):
        self.next_element = self.build_iterator(generator)

    def build_iterator(self, pair_gen: PairGenerator):
        batch_size = 10
        prefetch_batch_buffer = 5

        dataset = tf.data.Dataset.from_generator(pair_gen.get_next_pair,
                                                 output_types={PairGenerator.person1: tf.string,
                                                               PairGenerator.person2: tf.string,
                                                               PairGenerator.label: tf.bool})
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        return Inputs(element[self.img1_resized],
                      element[self.img2_resized],
                      element[PairGenerator.label])

    def _read_image_and_resize(self, pair_element):
        target_size = [128, 128]
        # read images from disk
        img1_file = tf.read_file(pair_element[PairGenerator.person1])
        img2_file = tf.read_file(pair_element[PairGenerator.person2])
        img1 = tf.image.decode_image(img1_file)
        img2 = tf.image.decode_image(img2_file)

        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img1.set_shape([None, None, 3])
        img2.set_shape([None, None, 3])

        # resize to model input size
        img1_resized = tf.image.resize_images(img1, target_size)
        img2_resized = tf.image.resize_images(img2, target_size)

        pair_element[self.img1_resized] = img1_resized
        pair_element[self.img2_resized] = img2_resized
        pair_element[self.label] = tf.cast(pair_element[PairGenerator.label], tf.float32)

        return pair_element
