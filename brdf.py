#import tensorflow as tf
import numpy as np
import h5py
import scipy
#from spline import linear_interpolate_1d

EPSILON = 1e-7
NEGATIVE = -1e-7

INTERPOLATE = 'piecewise_linear'

def spline_brdf(normals, light_directions, parameters=None, n_knots=90, n_channels=3, normalise=True, order=2):
    with tf.name_scope('brdf'):
        train_points, train_values = parameters
        train_points = tf.concat([train_points, tf.constant([-EPSILON, -1], dtype=train_points.dtype)], axis=0)
        train_values = tf.concat([train_values, tf.reshape(tf.constant([-EPSILON]*n_channels+[NEGATIVE]*n_channels, dtype=train_values.dtype), (2,n_channels))], axis=0)
        n_knots = train_points.shape[0]

        train_points = tf.reshape(train_points, [1, n_knots, 1])
        train_values = tf.reshape(tf.convert_to_tensor(train_values), [1, n_knots, n_channels])

        if normalise:
            normals = normals / tf.norm(normals, axis=-1, keepdims=True) 
            light_directions = light_directions / tf.norm(light_directions, axis=-1, keepdims=True)

        query_points = tf.reduce_sum(normals * light_directions, axis=-1)


        
        query_points = tf.reshape(query_points, [1, -1, 1])
        
        if INTERPOLATE == 'piecewise_linear':
            train_points = tf.reshape(train_points, [-1])
            train_values = tf.reshape(train_values, [-1, n_channels])

            query_points = tf.reshape(query_points, [-1])
            query_values =  linear_interpolate_1d(train_points, train_values, query_points)

        else:

            query_values = tf.contrib.image.interpolate_spline(
                train_points,
                train_values,
                query_points,
                order,
                regularization_weight=0.0,
                name='interpolate_spline'
            )

            query_values = tf.reshape(query_values, [-1, n_channels])
    
        return query_values


def interpolate_brdf_np(normals, light_directions, parameters=None, n_knots=90, n_channels=3, normalise=True, order=2):
    train_points, train_values = parameters
    train_points = np.concatenate([train_points, [-EPSILON, -1]], axis=0)
    train_values = np.concatenate([train_values, np.reshape([-EPSILON]*n_channels+[NEGATIVE]*n_channels, (2,n_channels))], axis=0)
    n_knots = train_points.shape[0]

    train_points = np.reshape(train_points, [n_knots])
    train_values = np.reshape(train_values, [n_knots, n_channels])

    if normalise:
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True) 
        light_directions = light_directions / np.linalg.norm(light_directions, axis=-1, keepdims=True)

    query_points = np.sum(normals * light_directions, axis=-1)

    f = scipy.interpolate.interp1d(train_points, train_values, axis=0)

    try:
        query_values = f(np.clip(query_points.flatten(), -1, 1))
    except ValueError as ve:
        print(query_points.max()<=1, query_points.min()>=-1)
        print(query_points.max(), query_points.min())
        print(train_points.max(), train_points.min())

        raise ve

    return query_values

def create_trainable_knots(no_knots, no_channels, dtype):
    train_points = tf.cast(tf.linspace(0.0, 1.0, no_knots), dtype) * np.pi / 2
    train_points = tf.cos(train_points)
    train_values = tf.Variable(tf.stack([train_points]*3,axis=-1))
    #train_values = tf.get_variable("knots_values", [no_knots, no_channels], dtype=dtype, initializer=tf.initializers.random_normal, trainable=True)
    return train_points, train_values


def create_constant_knots_from_merl(merl_path, dtype, no_knots=None):
    with h5py.File(merl_path, 'r') as f:
        samples = f['BRDF'][:,:,0,0] # [3, 90]
        #samples = np.stack([samples[0], samples[0], samples[0]], axis=0) # [3, 90]
        #samples = np.minimum(samples, 0.11)
    train_values = samples.T
    no_samples = train_values.shape[0]
    train_points = np.cos(np.arange(no_samples) * np.pi / 180)
    if no_knots is not None:
        f = scipy.interpolate.interp1d(train_points, train_values, 'linear', axis=0)
        train_points = np.cos(np.linspace(0, 88.9, no_knots) * np.pi / 180)
        train_values = f(train_points)
    
    return train_points.astype(dtype), train_values.astype(dtype)

def create_constant_knots_lambert(dtype):
    no_samples = 90
    train_points = np.cos(np.arange(no_samples) * np.pi / 180)
    return train_points.astype(dtype), np.stack([(train_points-0.5)**2]*3,axis=-1).astype(dtype)


