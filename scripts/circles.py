# unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with unet.  If not, see <http://www.gnu.org/licenses/>.
import logging

import numpy as np

from tensorflow.python import ipu

import sys
sys.path.append('../src/')

import unet
from unet.datasets import circles

LEARNING_RATE = 1e-3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
np.random.seed(98765)


def train():
    train_dataset, validation_dataset = circles.load_data(1000, nx=256, ny=256, splits=(0.8, 0.2))
    train_dataset = train_dataset.repeat(10)
    validation_dataset = validation_dataset.repeat(10)

    cfg = ipu.config.IPUConfig()
    # Request 1 IPU
    cfg.auto_select_ipus = 1
    # Apply the configuration
    cfg.configure_ipu_system()

    with ipu.ipu_strategy.IPUStrategy(enable_dataset_iterators=False).scope():
        unet_model = unet.build_model(channels=circles.channels,
                                    num_classes=circles.classes,
                                    layer_depth=3,
                                    filters_root=16)
        unet.finalize_model(unet_model)
        
        trainer = unet.Trainer(checkpoint_callback=False)    
        trainer.fit(unet_model,
                    train_dataset,
                    validation_dataset,
                    epochs=5,
                    batch_size=2)
    return unet_model


if __name__ == '__main__':
    train()
