"""
This module prepares a 3D MRI scan data volume with T1w channel
for multi-class gmentation using FPMMID model
"""

import os
import logging
import numpy as np
from modules import vol_reader as vlr

class Subject:
    """
    This class provides the data preparations step for MRI scan volumes
    """
    def __init__(self, name):
        self.name = name
        self.num_channels = 1
        self.num_classes = 4
        self.data_type = "nii.gz"

    def prep_data(self, input_path, root_dir):
        """
        This method converts the mri scan to the input shape needed for
        the neural network
        """
        # adding the log file
        logging.basicConfig(filename = os.path.join(root_dir, "pred.log"), \
                            filemode = 'a',level = logging.INFO, \
                            format='%(levelname)s:%(message)s')
        # Find the dimensions of the imported volume
        logging.info("Data Prep - Started...")
        vol = vlr.volume_reader(os.path.join(input_path))
        vol /= np.max(vol)
        dims = list(vol.shape)
        vol = vol.reshape(1, dims[0], dims[1], dims[2], self.num_channels)
        sid = self.find_id(input_path)
        logging.info("Data Prep - Finished")
        return dims, self.num_channels, self.num_classes, sid, vol

    @staticmethod
    def find_id(input_path):
        """
        This method will assign a subject id to the input scan
        """
        sid = input_path.split("/")[-1]
        sid = sid.split("_")[0]
        return sid
