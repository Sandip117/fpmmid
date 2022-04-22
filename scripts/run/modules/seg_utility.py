"""
This module provides utilities for pre-processing and post-processing of segmented images
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as ks
from sklearn.preprocessing import LabelEncoder

COLORM = 'jet'

def get_args():
    """
    Get user input arguments and parse them
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path','-i', \
						help='Directory path where ping data exist')
    parser.add_argument('--out_dir','-o', \
						help='Directory path where ping data exist')
    parser.add_argument('--root_dir','-r', \
						help='root directory where the project exists')
    args = parser.parse_args()
    input_path = args.input_path
    out_dir = args.out_dir
    root_dir = args.root_dir

    return input_path, out_dir, root_dir

def class_weight(y_train, num_classes):
    """
    calculate the class weights based on the frequency of different classes
    """
    # Make a one dimensional long array - just get one sample volume to save
    y_train_mod = np.argmax(y_train, axis = 4).reshape(-1,1)
    class_freq = {}
    for i in y_train_mod:
        if i[0] in class_freq: # i is a 1d array, [0] will get the element
            class_freq[i[0]] += 1
        else:
            class_freq[i[0]] = 1
    for i in class_freq:
        class_freq[i] = len(y_train_mod) / (num_classes * class_freq[i])
    # Balanced Weights: wj=n_samples / (n_classes * n_samplesj)
    # https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/
    return class_freq

def mask_show(volume, mask_p, mask_p_path, data_type):
    """
    This function creates a comparison diagram for mid-z slice of
    input MR scan and predicted segmentation
    """
    mask_pred_comb = np.argmax(mask_p, axis = 3)
    mask_pred_slice = mask_pred_comb[int(mask_pred_comb.shape[0] / 2), :, :]
    fig, axs = plt.subplots(1,2, figsize=(10,8))
    fig.suptitle("Multi-Class Semantic Segmentation: FPMMID - "\
                 + str(data_type))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[1].imshow(mask_pred_slice, cmap = COLORM)
    axs[0].imshow(volume[int(volume.shape[0] / 2), :, :, 0], cmap = COLORM)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[1].set_title('WM, GM, CSF - AI Prediction')
    axs[0].set_title('T1w')
    fig.savefig(mask_p_path)

def mask_pred(volume, mask_act, mask_p, mask_p_path, data_type):
    """
    This function creates a comparison diagram for mid-z slice of
    input MR scan, ground truth segmentation, and predicted segmentation
    """
    mask_act_comb = np.argmax(mask_act, axis = 3)
    mask_act_slice = mask_act_comb[int(mask_act_comb.shape[0] / 2), :, :]
    mask_pred_comb = np.argmax(mask_p, axis = 3)
    mask_pred_slice = mask_pred_comb[int(mask_pred_comb.shape[0] / 2), :, :]
    fig, axs = plt.subplots(1,3, figsize=(10,8))
    fig.suptitle("Multi-Class Semantic Segmentation: MRI PING - "\
                 + str(data_type))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[1].imshow(mask_act_slice, cmap = COLORM)
    axs[2].imshow(mask_pred_slice, cmap = COLORM)
    axs[0].imshow(volume[int(volume.shape[0] / 2), :, :, 0], cmap = COLORM)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[1].set_title('WM, GM, CSF - Ground Truth')
    axs[2].set_title('WM, GM, CSF - AI Prediction')
    axs[0].set_title('T1w')
    fig.savefig(mask_p_path)
    print("volume and masks (actual and predicted) were written to: {}\n"\
          .format(mask_p_path))

def mask_pred_spatial(volume, mask_act, mask_p, mask_p_path, data_type):
    """
    This function creates slices of the model's final prediction
    for all z slices
    """
    mask_act_comb = np.argmax(mask_act, axis = 3)
    mask_pred_comb = np.argmax(mask_p, axis = 3)
    for z_index in range(0, int(volume.shape[0])):
        fig, axs = plt.subplots(1,3, figsize=(10,8))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.suptitle("Multi-Class Semantic Segmentation: MRI PING - " \
                  + str(data_type) + \
					 "\n Spatial Slices: slice_index = " + str(z_index))
        mask_act_slice = mask_act_comb[z_index, :, :]
        mask_pred_slice = mask_pred_comb[z_index, :, :]
        axs[1].imshow(mask_act_slice, cmap = COLORM)
        axs[2].imshow(mask_pred_slice, cmap = COLORM)
        axs[0].imshow(volume[z_index, :, :, 0], cmap = COLORM)
        for i in range(0,3):
            axs[i].axis('off')
            axs[1].set_title('WM, GM, CSF - Ground Truth')
            axs[2].set_title('WM, GM, CSF - AI Prediction')
            axs[0].set_title('T1')
            fig.savefig(mask_p_path + "/" + str(z_index) + ".png")

def mask_pred_temporal(volume, mask_act, mask_p, mask_p_path, data_type,\
                       epoch, elapsed):
    """
    This function creates timestamp mid-z slices of the model's prediction
    based on consecutive checkpoints until convergence
    """
    mask_act_comb = np.argmax(mask_act, axis = 3)
    mask_pred_comb = np.argmax(mask_p, axis = 3)
    mask_act_slice = mask_act_comb[int(mask_act_comb.shape[0] / 2), :, :]
    mask_pred_comb = np.argmax(mask_p, axis = 3)
    mask_pred_slice = mask_pred_comb[int(mask_pred_comb.shape[0] / 2), :, :]
    fig, axs = plt.subplots(1,3, figsize=(10,8))
    fig.suptitle("Multi-Class Semantic Segmentation: MRI PING - " \
               + str(data_type) + \
					"\n Temporal Convergence: Elapsed Time = {:.1f}". \
                        format(elapsed) + " minutes")
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[1].imshow(mask_act_slice, cmap = COLORM)
    axs[2].imshow(mask_pred_slice, cmap = COLORM)
    axs[0].imshow(volume[int(volume.shape[0] / 2), :, :, 0], cmap = COLORM)
    for i in range(0,3):
        axs[i].axis('off')
        axs[1].set_title('WM, GM, CSF - Ground Truth')
        axs[2].set_title('WM, GM, CSF - AI Prediction')
        axs[0].set_title('T1')
        fig.savefig(mask_p_path + "/" + str(epoch) + ".png")

def mask_sample(parsed_tfrec_dataset, mask_sample_path):
    """
    This function creates a sample slice image out of the
    parsed tfrecords for a given MRI volume
    """
    for volume_features in parsed_tfrec_dataset.take(1):
        mask = volume_features['label']
        mask = tf.io.parse_tensor(mask, out_type = tf.float32)
        mask_comb = np.argmax(mask, axis=3)
        mask_slice = mask_comb[int(mask_comb.shape[0] / 2), :, :]
        volume = volume_features['volume']
        volume = tf.io.parse_tensor(volume, out_type = tf.float32)
        fig, axs = plt.subplots(1,2, figsize=(10,8))
        fig.suptitle("Multi-Class Semantic Segmentation: MRI PING")
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        axs[1].imshow(mask_slice, cmap = COLORM)
        axs[0].imshow(volume[int(volume.shape[0] / 2), :, :, 0], cmap = COLORM)
        axs[1].axis('off')
        axs[0].axis('off')
        axs[1].set_title('Ground Truth')
        axs[0].set_title('T1w')
        fig.savefig(mask_sample_path)
        print("data parsed from tfrecord and sample mask written to: {}\n" \
        .format(mask_sample_path))

def scaling_plot(csv_path,plot_path):
    """
    Creating time-scaling plots with different number of GPU workers
    """
    # adding the log file
    scaling_data = pd.read_csv(csv_path)
    num_gpus = scaling_data['num_gpus']
    proc_time = scaling_data['proc_time']
    train_loss = scaling_data['loss']
    train_dice = scaling_data['dice_coef']
    train_meaniou = scaling_data['mean_iou']
    val_loss = scaling_data['val_loss']
    val_dice = scaling_data['val_dice_coef']
    val_meaniou = scaling_data['val_mean_iou']
    fig, axs = plt.subplots(2,2, figsize=(12,11))
    axs[0,0].bar(num_gpus, proc_time)
    axs[0,1].bar(num_gpus, train_loss)
    axs[1,0].bar(num_gpus, train_dice)
    axs[1,1].bar(num_gpus, train_meaniou)
    axs[0,0].set_title('Training Processing Time')
    axs[0,1].set_title('Training Loss')
    axs[1,0].set_title('Training Dice Coefficient')
    axs[1,1].set_title('Training Mean IOU')
    count = 0
    for ax_ in axs.flat:
        if count == 0:
            ax_.set(xlabel='Number of GPUs', ylabel='Processing Time (Sec)')
        elif count == 1:
            ax_.set(xlabel='Number of GPUs', ylabel='Loss')
        elif count == 2:
            ax_.set(xlabel='Number of GPUs', ylabel='Dice Coefficient')
        elif count == 3:
            ax_.set(xlabel='Number of GPUs', ylabel='Mean IOU')
        count += 1
        ax_.set(xticks=[1, 2, 3, 4])
    plot_path_train = os.path.join(plot_path,'training_scaling.png')
    fig.savefig(plot_path_train)
    print("Figure saved in: {}".format(plot_path_train))

    fig, axs = plt.subplots(2,2, figsize=(12,11))
    axs[0,0].bar(num_gpus, proc_time)
    axs[0,1].bar(num_gpus, val_loss)
    axs[1,0].bar(num_gpus, val_dice)
    axs[1,1].bar(num_gpus, val_meaniou)
    axs[0,0].set_title('Validation Processing Time')
    axs[0,1].set_title('Validation Loss')
    axs[1,0].set_title('Validation Dice Coefficient')
    axs[1,1].set_title('Validation Mean IOU')
    count = 0
    for ax_ in axs.flat:
        if count == 0:
            ax_.set(xlabel='Number of GPUs', ylabel='Processing Time (Sec)')
        elif count == 1:
            ax_.set(xlabel='Number of GPUs', ylabel='Loss')
        elif count == 2:
            ax_.set(xlabel='Number of GPUs', ylabel='Dice Coefficient')
        elif count == 3:
            ax_.set(xlabel='Number of GPUs', ylabel='Mean IOU')
        count += 1
        ax_.set(xticks=[1, 2, 3, 4])
    plot_path_valid = os.path.join(plot_path,'valid_scaling.png')
    fig.savefig(plot_path_valid)
    print("Figure saved in: {}".format(plot_path_valid))

def mask_encoder(y_train, num_classes):
    """
    This function encodes labels to a sequential order starting with
    label 0,1,2...
    """
    y_train_shape = y_train.shape
    # Make a one dimensional long array
    y_train_mod = y_train.reshape(-1,1)
    # Encode all entries in sequential mode, i.e., 1,2,4,5 would
    # become 1,2,3,4 for instance
    y_train_mod = LabelEncoder().fit_transform(y_train_mod)
    y_train_mod = y_train_mod.reshape(y_train_shape)
    y_train_cat = ks.utils.to_categorical(y_train_mod, \
                                          num_classes = num_classes)
    return y_train_cat
