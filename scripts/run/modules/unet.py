"""
This module provides the 3D UNET Architecture with additional
regularization methods, and loss functions
"""
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_addons as tfa
import horovod.tensorflow.keras as hvd

eps = ks.backend.epsilon()
PADDING = 'same'
KERNEL_SIZE = (3,3,3)

# UNET Architecture
def network(inputs, num_classes, d_rate, init_fil, scale, activation1):
    """
    This function will create the Neural network for training and inference
    """
    gn_ = 1 # should be 4 - testing now
    gn_init = inputs.shape[-1]
	# Ensure gn_ <= number of input channels
    # number of layers for group normalization cannot ex_midceed number of
    gn_init = min(gn_init, gn_)
    x_mid = tfa.layers.GroupNormalization(groups=gn_init, axis = -1)(inputs)
	# two layers to with init_fil filters and same size of images
    x_mid = ks.layers.Conv3D(filters = init_fil // scale,\
                             kernel_size = KERNEL_SIZE,\
                             strides = (1,1,1), \
                             activation = activation1,\
                             padding = PADDING)(x_mid)

    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(filters = init_fil // scale,\
                             kernel_size = KERNEL_SIZE,\
                             strides = (1,1,1), padding = PADDING,\
                             activation = activation1)(x_mid)

    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    lx_0 = x_mid # for the skipping step

    x_mid = ks.layers.MaxPooling3D(pool_size = (2,2,2),\
                                   strides = (2,2,2),\
                                   padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)

    x_mid = ks.layers.Conv3D(init_fil * 2 // scale, \
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)

    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 2 // scale, \
                             KERNEL_SIZE, padding = PADDING, \
                             activation = activation1)(x_mid)

    x_mid = ks.layers.Dropout(d_rate)(x_mid)

    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    lx_1 = x_mid

    x_mid = ks.layers.MaxPooling3D(pool_size = (2,2,2),\
                                   strides = (2,2,2),\
                                   padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)

    x_mid = ks.layers.Conv3D(init_fil * 4 // scale,\
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)

    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 4 // scale,\
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)

    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    lx_2 = x_mid
    x_mid = ks.layers.MaxPooling3D(pool_size = (2,2,2),\
                                   strides = (2,2,2),\
                                   padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 8 // scale,\
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 8 // scale,\
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    lx_3 = x_mid
    x_mid = ks.layers.MaxPooling3D(pool_size = (2,2,2),\
                                   strides = (2,2,2),\
                                   padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 16 // scale,\
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 16 // scale,\
                             KERNEL_SIZE, padding = PADDING,\
                             activation = activation1)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3DTranspose(filters = init_fil * 8 // scale,\
                                      kernel_size = KERNEL_SIZE,\
                                      strides = (2,2,2),\
                                      activation = activation1,\
                                      padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Concatenate()([lx_3, x_mid])
    # note = conv2d transpose vs upsampling\n conv2d transpose: 
    # For each pix_midel of input,it applies the kernel -multiply
    # pix_midels by kernel weights and creates a 3x_mid3 subset in this case,
    # then march through the other pix_midels thus it's both applying 
    # a kernel and increasing the size upsampling2d: for each pix_midel
    # of the input, it just copies the pix_midel and creates a 3x_mid3 subset
    # in this case. so no applying of kernels, More info:
    # https://www.youtube.com/watch?v=ilkSwsggSNM)
    x_mid = ks.layers.Conv3D(init_fil * 8 // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3DTranspose(filters = init_fil * 4 // scale,\
                                      kernel_size = KERNEL_SIZE,\
                                      strides = (2,2,2),\
                                      activation = activation1,\
                                      padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Concatenate()([x_mid , lx_2])
    x_mid = ks.layers.Conv3D(init_fil * 4 // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 4 // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3DTranspose(init_fil * 2 // scale,\
                                      KERNEL_SIZE, strides = (2,2,2),\
                                      activation = activation1,\
                                      padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Concatenate()([x_mid , lx_1])
    x_mid = ks.layers.Conv3D(init_fil * 2 // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)

    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil * 2 // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)

    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3DTranspose(init_fil // scale,\
                                      KERNEL_SIZE, strides = (2,2,2),\
                                      activation = activation1,\
                                      padding = PADDING)(x_mid)

    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Concatenate()([x_mid , lx_0])
    x_mid = ks.layers.Conv3D(init_fil // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    x_mid = ks.layers.Conv3D(init_fil // scale,\
                             KERNEL_SIZE, activation = activation1,\
                             padding = PADDING)(x_mid)
    x_mid = ks.layers.Dropout(d_rate)(x_mid)
    x_mid = tfa.layers.GroupNormalization(groups=gn_, axis = -1)(x_mid)
    outputs = ks.layers.Conv3D(num_classes, (1,1,1), activation = "softmax")(x_mid)
    # no normalization/dropout on the output layer
	# To match the mask sizes
	#outputs = ks.layers.Cropping3D(cropping = ((3,2),(0,0),(0,0)))(outputs)
    return outputs
### end of UNET ARCH ####

def dice_coef(y_true, y_pred):
    """
    Dice coefficient similarlity metrics
    """
    y_true_f = ks.backend.flatten(y_true[:,:,:,:,1:]) # ign_ore background layer
    y_pred_f = ks.backend.flatten(y_pred[:,:,:,:,1:]) # ign_ore background layer
    intersection = ks.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (ks.backend.sum(y_true_f) \
            + ks.backend.sum(y_pred_f) + eps)

def dice_coef_loss(y_true, y_pred):
    """
    Dice coefficient loss function
    """
    return 1 - dice_coef(y_true, y_pred)

def w_dice_coef_loss(sample_weight):
    """
    Class-weighted Dice coefficient loss function
    """
    def w_dice_coef_loss_fun(y_true, y_pred):
        dims = y_true.shape
        intersection = eps
        denom = eps
        for i in range(0, dims[4]):
            y_true_f = ks.backend.flatten(y_true[:,:,:,:,i])
            y_pred_f = ks.backend.flatten(y_pred[:,:,:,:,i])
            intersection += sample_weight[i] * ks.backend.sum(y_true_f * y_pred_f)
            denom += sample_weight[i] * ks.backend.sum(y_true_f + y_pred_f)
        coef = 2. * intersection / denom
        # internal return will be called within the neural network training step
        return 1 - coef
    return w_dice_coef_loss_fun

def w_dice_coef(sample_weight):
    """
    Class-weighted Dice similarity metrics
    """
    def w_dice_coef_fun(y_true, y_pred):
        dims = y_true.shape
        intersection = eps
        denom = eps
        for i in range(0, dims[4]):
            y_true_f = ks.backend.flatten(y_true[:,:,:,:,i])
            y_pred_f = ks.backend.flatten(y_pred[:,:,:,:,i])
            intersection += sample_weight[i] * ks.backend.sum(y_true_f * y_pred_f)
            denom += sample_weight[i] * ks.backend.sum(y_true_f + y_pred_f)
        coef = 2. * intersection / denom
        return coef
    return w_dice_coef_fun

def ccw(sample_weight):
    """
    Weighted categorical cross entropy loss function
    sample_weights are fed into the core loss function
    """
    def ccw_loss(y_true, y_pred):
        """
        Weighted Categorical cross entropy loss function
        core loss function  with sample_weights fed from the upstream function
        """
        dims = y_true.shape
        y_true_f = ks.backend.reshape(y_true, (dims[0] * dims[1] * dims[2] * dims[3], dims[4]))
        y_pred_f = ks.backend.reshape(y_pred, (dims[0] * dims[1] * dims[2] * dims[3], dims[4]))
        loss = eps

        # https://github.com/keras-team/keras/blob/d8fcb9d4d4dad45080ecfdd575483653028f8eda/keras/backend.py#L5039
        # scale preds so that the class probas of each sample sum to 1
        y_pred_f = y_pred_f / tf.reduce_sum(y_pred_f, -1, True)

        # Compute cross entropy from probabilities.
        epsilon_ = tf.constant(ks.backend.epsilon(), y_pred_f.dtype.base_dtype)
        y_pred_f = tf.clip_by_value(y_pred_f, epsilon_, 1. - epsilon_)

        for i in range(0, dims[4]):
            loss += - sample_weight[i] * ks.backend.sum(y_true_f[:, i] \
                    * tf.math.log(y_pred_f[:, i]))
        loss /= (dims[0] * dims[1] * dims[2] * dims[3] * dims[4])
        return loss
    return ccw_loss

def mean_iou_argmax(y_true, y_pred):
    """
    intersection over union similarity metrics
    """
    y_pred_argmax = tf.keras.backend.argmax(y_pred, axis = -1)
    y_true_argmax = tf.keras.backend.argmax(y_true, axis = -1)
    mio = tf.keras.metrics.MeanIoU(num_classes=4) # consider feeding num_classes as
                                                # a function-in-function similar to ccw
    mio.update_state(y_true_argmax, y_pred_argmax)
    return mio.result().numpy()

def create_model(init_lr, inputs, is_parallel, outputs, sample_weight):
    """
    Create Neural Network model
    """
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    if is_parallel == "true":
        opt = tf.keras.optimizers.Adam(learning_rate = init_lr * hvd.size())
        opt = hvd.DistributedOptimizer(opt)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate = init_lr)

    model.compile(optimizer = opt, loss = w_dice_coef_loss(sample_weight = sample_weight) ,\
                  metrics =[w_dice_coef(sample_weight = sample_weight), dice_coef,\
                  mean_iou_argmax], experimental_run_tf_function=False, run_eagerly=True)
    return model
