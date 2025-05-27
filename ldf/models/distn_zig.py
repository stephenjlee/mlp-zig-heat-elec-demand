import json

# noinspection PyUnresolvedReferences
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import scipy
from scipy.stats import gamma

from ldf.models.distn_base import DistnBase


def softplus(x):
    return np.log1p(np.exp(x))

class DistnZIG(DistnBase):

    def __init__(self, params):
        super().__init__(params)
        self.epsilon = np.finfo(np.float32).eps  # Small constant to avoid numerical issues

    def get_distn_type(self):
        return 'mixed'  # Since ZIG is a mixture of discrete and continuous components

    @staticmethod
    def np_ll(x, p_zero, shape, scale):
        """
        Compute the log-likelihood using NumPy for the ZIG distribution.
        """

        # Ensure parameters are valid
        p_zero = np.clip(p_zero, (1e-7), (1. - 1e-7))
        shape = np.maximum(shape, 1e-7)
        scale = np.maximum(scale, 1e-7)

        # Initialize log-likelihood array
        ll = np.zeros_like(x, dtype=np.float64)

        # Log-likelihood for zeros
        mask_zero = x == 0
        ll[mask_zero] = np.log(p_zero[mask_zero])

        # Log-likelihood for positive values
        mask_positive = x > 0
        ll[mask_positive] = np.log(1 - p_zero[mask_positive]) + gamma.logpdf(x[mask_positive], a=shape[mask_positive], scale=scale[mask_positive])

        return ll


    def tf_nll(self, y_true, y_pred):

        """
        Compute the negative log-likelihood using TensorFlow for training.
        This version handles numerical instability robustly.
        """

        # Transform parameters
        p = tf.clip_by_value(tf.nn.sigmoid(y_pred[:, 0]), 1e-7, 1 - 1e-7)
        shape = tf.maximum(tf.nn.softplus(y_pred[:, 1]), 1e-7)
        scale = tf.maximum(tf.nn.softplus(y_pred[:, 2]), 1e-7)

        # Flatten tensors
        p = K.flatten(p)
        shape = K.flatten(shape)
        scale = K.flatten(scale)
        y_true = K.flatten(y_true)

        # Indicators for zero and positive observations
        is_zero = K.cast(K.equal(y_true, 0), tf.float32)

        # this is now a "soft mask"
        is_positive = K.cast(K.greater(y_true, 0), tf.float32) + 1e-7  # Add smoothing

        # Compute log-likelihood components
        log_gamma_pdf = tfp.distributions.Gamma(concentration=shape, rate=1.0 / scale).log_prob(
            tf.maximum(y_true, 1e-7)
        )
        log_gamma_pdf = tf.where(tf.math.is_finite(log_gamma_pdf), log_gamma_pdf, tf.zeros_like(log_gamma_pdf))

        ll_zero = tf.math.log(p) * is_zero

        clamped_1_minus_p = tf.clip_by_value(1 - p, 1e-7, 1.0)

        ll_positive = is_positive * (tf.math.log(clamped_1_minus_p) + log_gamma_pdf)

        # Combine log-likelihood components
        log_likelihood = ll_zero + ll_positive

        # Negative log-likelihood
        nll = -tf.reduce_mean(log_likelihood)

        return nll


    def interpret_predict_output(self, yhat):
        """
        Interpret the output predictions from the model.
        """
        p_zero = yhat[:, 0]
        shape = yhat[:, 1]
        scale = yhat[:, 2]

        # Ensure parameters are valid
        p_zero = np.clip(scipy.special.expit(p_zero), (1e-7), (1. - 1e-7))
        shape = np.maximum(softplus(shape), 1e-7)
        scale = np.maximum(softplus(scale), 1e-7)

        # Expected mean and standard deviation of the ZIG distribution
        mean_preds = (1 - p_zero) * shape * scale
        variance = (1 - p_zero) * (shape * scale ** 2 + p_zero * (shape * scale) ** 2)
        std_pred = np.sqrt(variance)

        # Prepare parameter dictionaries
        preds_params = {'p_zero': p_zero, 'shape': shape, 'scale': scale}
        preds_params_flat = [
            {'p_zero': p, 'shape': s, 'scale': sc}
            for p, s, sc in zip(p_zero.flatten(), shape.flatten(), scale.flatten())
        ]

        return mean_preds, std_pred, preds_params, preds_params_flat

    def compute_nll(self, preds_params, y):
        """
        Compute the negative log-likelihood for given predictions and true values.
        """
        p_zero = preds_params['p_zero']
        shape = preds_params['shape']
        scale = preds_params['scale']

        nlls = -1.0 * self.np_ll(y, p_zero, shape, scale)
        mean_nll = np.mean(nlls)
        return mean_nll, nlls

    def ppf(self, q, p_zero, shape, scale):
        """
        Percent point function (inverse CDF) for the ZIG distribution.
        """

        q = np.asarray(q)
        result = np.zeros_like(q, dtype=np.float64)

        # Adjust parameters
        p_zero = np.clip(p_zero, self.epsilon, 1 - self.epsilon)
        shape = np.maximum(shape, self.epsilon)
        scale = np.maximum(scale, self.epsilon)

        # Masks for zero and positive quantiles
        mask_zero = q <= p_zero
        mask_positive = q > p_zero

        # Compute PPF
        result[mask_zero] = 0

        if p_zero.size == 1:
            adjusted_q = (q[mask_positive] - p_zero) / (1 - p_zero)
            result[mask_positive] = gamma.ppf(adjusted_q, a=shape, scale=scale)
        else:
            adjusted_q = (q[mask_positive] - p_zero[mask_positive]) / (1 - p_zero[mask_positive])
            result[mask_positive] = gamma.ppf(adjusted_q, a=shape[mask_positive], scale=scale[mask_positive])

        return result

    def ppf_params(self, q, params):
        """
        PPF using parameter dictionary.
        """
        return self.ppf(q, params['p_zero'], params['shape'], params['scale'])

    def cdf(self, x, p_zero, shape, scale):
        """
        Cumulative distribution function for the ZIG distribution.
        """
        x = np.asarray(x)
        cdf_values = np.zeros_like(x, dtype=np.float64)

        # Adjust parameters
        p_zero = np.clip(p_zero, self.epsilon, 1 - self.epsilon)
        shape = np.maximum(shape, self.epsilon)
        scale = np.maximum(scale, self.epsilon)

        # Masks for different x ranges
        mask_neg = x < 0
        mask_zero = x == 0
        mask_positive = x > 0

        # Compute CDF
        cdf_values[mask_neg] = 0
        cdf_values[mask_zero] = p_zero
        gamma_cdf = gamma.cdf(x[mask_positive], a=shape, scale=scale)
        cdf_values[mask_positive] = p_zero + (1 - p_zero) * gamma_cdf

        return cdf_values

    def cdf_params(self, x, params):
        """
        CDF using parameter dictionary.
        """
        return self.cdf(x, params['p_zero'], params['shape'], params['scale'])

    def sample_posterior_params(self, params, size=None):
        """
        Sample from the posterior predictive distribution.
        """

        p_zero = params['p_zero']
        shape = params['shape']
        scale = params['scale']

        if size is None:
            size = p_zero.shape

        # Generate samples
        random_uniform = np.random.uniform(size=size)
        zeros = random_uniform < p_zero
        samples = np.zeros(size, dtype=np.float64)
        non_zero_indices = np.where(~zeros)

        # Sample from Gamma distribution for non-zero values
        samples[non_zero_indices] = gamma.rvs(
            a=shape[non_zero_indices],
            scale=scale[non_zero_indices],
            size=len(non_zero_indices[0])
        )

        return samples

    def get_output_dim(self):
        """
        Return the number of parameters output by the distribution.
        """
        return 3  # [p_zero, shape, scale]
