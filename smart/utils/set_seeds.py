import random
import numpy as np
import tensorflow as tf

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
