import os
import matplotlib.pyplot as plt
import vol_reader as vlr

base_path = "/project/RAD-Warfield/ai/comp_rad/arash/brats"
par_i = 1
scaling_dir = os.path.join(base_path, 'checkpoint-all', 'parallel', 'partition', str(par_i), '10-epoch')
scaling_data_path = os.path.join(scaling_dir,'scaling.csv')
plot_path = os.path.join(scaling_dir)
vlr.scaling_plot(scaling_data_path, plot_path)
