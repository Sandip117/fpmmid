"""
This module provides utilities for operations with tfrecords
including reading, writing, and parsing tfrecords
"""
import numpy as np
import tensorflow as tf

def _parse_function(example_proto):
    """
    parsing example proto into specified features
    """
    feature_description = {
       'volume': tf.io.FixedLenFeature([], tf.string, default_value=''),
       'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
       }
    return tf.io.parse_single_example(example_proto, feature_description)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(volume, label):
    """Returns serialize string example"""
    feature = {
        'volume': _bytes_feature(tf.io.serialize_tensor(volume)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
  }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(volume, label):
    """Creating an example proto out of the features in tfrecords"""
    tf_string = tf.py_function(
    serialize_example,
    (volume, label),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

def make_feature_of_bytes(tensor_serialized):
    """Converting features to bytes"""
    return tf.train.Feature(bytes_list= \
           tf.train.BytesList(value=[tensor_serialized.numpy()]))

def make_tf_example(volume, label):
    """Creating a full proto example"""
    # here labels are also tensors (segmented volumes)
    volume_serialized = tf.io.serialize_tensor(volume)
    label_serialized = tf.io.serialize_tensor(label)
    volume_feature_of_bytes = make_feature_of_bytes(volume_serialized)
    label_feature_of_bytes = make_feature_of_bytes(label_serialized)

    features_for_example = {'volume': volume_feature_of_bytes, 'label': label_feature_of_bytes}
    example_proto = tf.train.Example(features=\
                                     tf.train.Features(feature=features_for_example))\
                                     .SerializeToString()
    return example_proto

def get_tfrecord_size(parsed_data):
    """Get the number of records in the tfrecord file"""
    train_dataset_size = 0
    for records in parsed_data:
        train_dataset_size += 1
    return train_dataset_size

def get_data_unit_attr(parsed_data):
    """Get the data attributes for parsed tfrecords"""
    for volume_features in parsed_data.take(1):
        volume = volume_features['volume']
        volume = tf.io.parse_tensor(volume, out_type = tf.float32)
    return volume.shape[0:3], volume.shape[-1]

def get_dataset_batches(parsed_data, num_samples, dims, num_channels, num_classes):
    """Get a batch of records from tfrecords file"""
    volume_tot = np.zeros((num_samples, dims[0], dims[1], dims[2], num_channels),\
                           dtype = np.float32)
    label_tot = np.zeros((num_samples, dims[0], dims[1], dims[2], num_classes), dtype = np.float32)
    index = 0
    for volume_features in parsed_data.take(num_samples):
        volume = volume_features['volume']
        volume = tf.io.parse_tensor(volume, out_type = tf.float32)
        label = volume_features['label']
        label = tf.io.parse_tensor(label, out_type = tf.float32)
        volume_tot[index] = volume
        label_tot[index] = label
        index += 1
    return volume_tot, label_tot
