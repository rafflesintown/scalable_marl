import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import random, grad, make_jaxpr, jit, vmap
import optax
import itertools
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from tensorboardX import SummaryWriter
from datetime import datetime


import os
import argparse


# Define ELU activation
def elu(x):
    return jnp.where(x > 0, x, jnp.exp(x) - 1)


def init_rf(key, s_dim, rf_num, sigma=0.05, learn_rf=False):
    keys = random.split(key, num=2)  # Split key for different components

    # Fourier feature layer
    w_init_scale = 1.0 / sigma if sigma > 0 else 1.0
    fourier_w = (
        random.normal(keys[0], (s_dim, rf_num), dtype=jnp.float32) * w_init_scale
    )
    fourier_b = random.uniform(keys[1], (rf_num,), minval=0, maxval=2 * jnp.pi)

    return fourier_w, fourier_b


def init_params(key, rf_num):
    # Output layer initialization (weights)
    output = random.normal(key, (rf_num, 1))

    params = {
        "output": output,
    }

    return params


def layernorm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta


def forward(params, fourier_w, fourier_b, x):
    x = jnp.cos(jnp.dot(x, fourier_w) + fourier_b)

    x = jnp.dot(x, params["output"])
    return x


class algorithm:
    def __init__(
        self,
        env=None,
        N=10,
        kappa=2,
        trunc_kappa=1,
        s_dim=1,
        K0=None,
        m=200,
        gamma=0.9,
        n_epochs=10,
        eta=1e-3,
        use_true_grad=False,
        equal_weights=False,
        normalize_grad=False,
        use_nys=False,
        data_eps=10,
        key_int=0,
        alpha=3.0,
        noise_sig=1.0,
    ):
        self.env = env
        self.kappa = kappa
        self.trunc_kappa = trunc_kappa
        self.N = N
        self.K = K0 if K0 is not None else jnp.zeros((N, N))
        self.s_dim = s_dim
        self.m = m  # dimension of rf
        self.gamma = gamma
        self.n_epochs = n_epochs  # epochs
        self.e = 0  # current epoch
        self.key_int = key_int
        self.key = random.PRNGKey(self.key_int)
        self.keys = random.split(self.key, self.n_epochs)
        self.noise_sig = noise_sig
        omega, b = self.gen_rf()
        self.omega = omega
        self.b = b
        self.omega_central, self.b_central = self.gen_rf_central()
        if use_nys == True:
            self.nys_samples = self.get_nys_samples()
            self.use_nys = True
            self.gen_nystrom_central()
        else:
            self.use_nys = False
        self.theta = jnp.zeros((self.N, m))
        self.theta_central = jnp.zeros((m))
        self.optimizer = optax.adabelief(eta)
        self.opt_params = self.optimizer.init(self.K)
        self.eta = eta
        self.use_true_grad = use_true_grad
        self.equal_weights = equal_weights  # flag for whether sampling across steps in an episode should be uniform or weighted by discount factor
        self.normalize_grad = normalize_grad
        self.data_eps = data_eps
        self.alpha = alpha  # weight for the state cost (vs action cost)
        # self.critics_params = [init_params(random.PRNGKey(i), self.input_dim, self.hidden_dim, self.output_dim) for i in range(N)]
        # self.central_critics_params = init_params(random.PRNGKey(0), N, self.hidden_dim, self.output_dim)
        self.fourier_w, self.fourier_b = init_rf(self.key, N, m, noise_sig)
        self.rf_params = init_params(self.key, m)
        self.rf_optimizer = optax.adam(1e-4)
        self.opt_state = self.rf_optimizer.init(self.rf_params)

    def gen_rf(self):
        # Generate random features using JAX
        key_omega, key_b = random.split(self.key, 2)
        omega = (
            1.0
            / self.noise_sig
            * random.normal(
                key_omega, (self.N, (2 * self.kappa + 1) * self.s_dim, self.m)
            )
        )
        b = random.uniform(key_b, (self.N, self.m), minval=0, maxval=2 * jnp.pi)
        return (omega, b)

    def gen_rf_central(self):
        # Generate random features using JAX
        key_omega, key_b = random.split(self.key, 2)
        omega = random.normal(key_omega, (self.N * self.s_dim, self.m))
        b = random.uniform(key_b, (self.m,), minval=0, maxval=2 * jnp.pi)
        return (omega, b)

    # generates samples for nystrom
    def get_nys_samples(self, n_eps=50):
        f_s_data = [None for i in range(n_eps)]
        keys = random.split(self.key, n_eps)

        # Iterate over number of episodes to collect data
        for i in range(n_eps):
            # Run an episode
            _, _, f_s_data[i], _ = self.env.run_episode(
                K=self.K, s0=jnp.zeros(self.N), key=keys[i]
            )
        f_s_samples = self.get_samples(
            f_s_data, buffer_length=10, replace=False
        ).T  # should be array of size (n_eps * buffer_length,N)

        return f_s_samples

    def gen_nystrom_central(self):
        self.nys_samples = None
        if self.nys_samples is None:
            n_samples = self.m
            scale = 2
            samples = random.normal(self.key, (n_samples, self.N)) * scale
        samples_reshaped_1 = samples[:, jnp.newaxis, :]
        samples_reshaped_2 = samples[jnp.newaxis, :, :]
        diff = samples_reshaped_1 - samples_reshaped_2
        G = jnp.exp(-jnp.sum(samples_reshaped_1 - samples_reshaped_2, axis=2) ** 2 / 2)
        start = time.time()
        (D, V) = jnp.linalg.eig(G)
        print("eigendecomp took %.3f seconds" % (time.time() - start))
        self.nys_samples = samples  # n_samples by N
        self.nys_D = D
        self.nys_V = jnp.real(V)[:, : self.m]  # only take top m eigenvectors

    def get_features(self, f_s, i):
        f_s_cols = f_s.shape[1]
        f_s_double = jnp.concatenate([f_s, f_s], axis=1)

        f_s_i = f_s_double[
            :, jnp.r_[i - self.kappa : i + self.kappa + 1]
        ]  # see https://stackoverflow.com/questions/61497090/how-to-slice-starting-from-negative-to-a-positive-index-or-the-opposite

        phi_f_s_i = jnp.cos(jnp.dot(f_s_i, self.omega[i, :, :]) + self.b[i, :])
        return phi_f_s_i

    def get_features_central(self, f_s):
        phi_f_s = jnp.cos(jnp.dot(f_s, self.omega_central) + self.b_central)
        return phi_f_s

    def get_features_central_nys(self, f_s):
        f_s_reshaped = f_s[:, jnp.newaxis, :]
        nys_samples_reshaped = self.nys_samples[jnp.newaxis, :, :]
        diff = f_s_reshaped - nys_samples_reshaped
        k_f_s_vec = jnp.exp(
            -jnp.sum(diff, axis=2) ** 2 / 2.0
        )  # this is n_batch by self.m

        phi_f_s = k_f_s_vec @ self.nys_V

        return phi_f_s

    def policy_eval(self, f_s, f_s_next, r_next, n_opt=10):
        start = time.time()
        for i in range(self.N):
            phi_i = self.get_features(f_s, i)
            phi_next_i = self.get_features(f_s_next, i)
            r_next_i = r_next[:, i]

            LHS = jnp.dot(phi_i.T, phi_i) - self.gamma * jnp.dot(phi_i.T, phi_next_i)
            RHS = jnp.dot(phi_i.T, r_next_i)

            sol, _, _, _ = jnp.linalg.lstsq(LHS, RHS)
            # sol,_,_,_ = scipy.linalg.lstsq(LHS, RHS,lapack_driver='gelsy')
            self.theta = self.theta.at[i, :].set(sol)
        print("policy eval took %.3f seconds" % (time.time() - start))

    def policy_eval_central(self, phi, phi_next, r_next):
        start = time.time()
        LHS = jnp.dot(phi.T, phi) - self.gamma * jnp.dot(phi.T, phi_next)
        RHS = jnp.dot(phi.T, r_next)

        start = time.time()
        sol, _, _, _ = scipy.linalg.lstsq(LHS, RHS, lapack_driver="gelsy")
        print("Lhs shape", LHS.shape)
        print("RHS shape", RHS.shape)
        # print("so; shape", sol.shape)
        print("lstsq took %.3f seconds" % (time.time() - start))
        start = time.time()
        self.theta_central = self.theta_central.at[:].set(sol)
        print("setting theta took %.3f seconds" % (time.time() - start))

        print("LHS shape", LHS.shape)

    # new version of policy-update with batch
    def policy_update(self):
        start = time.time()
        s_samples = self.get_samples(
            s_data=self.s_data,
            buffer_length=10 * self.data_eps,
            equal_weights=self.equal_weights,
        )

        V_K_fn = lambda K: self.get_approx_V(K, s_samples=s_samples)
        start = time.time()
        grad_K = grad(V_K_fn)(self.K)

        true_grad_K = grad(self.get_true_VK)(self.K)
        if self.use_true_grad == True:
            grad_K = true_grad_K
        start_norm = time.time()

        if self.normalize_grad == True:
            grad_K = grad_K.at[:].set(grad_K / jnp.linalg.norm(grad_K, ord="fro"))
        start = time.time()
        self.K = self.K.at[:].set(self.K - self.eta * grad_K)
        self.K = self.trunc_kappa_controller(self.K)

    def policy_update_central(self):
        s_samples = self.get_samples(
            s_data=self.s_data,
            buffer_length=10 * self.data_eps,
            equal_weights=self.equal_weights,
        )
        n_samples = s_samples.shape[1]
        if self.use_nys == True:
            Q_A_fn = lambda a_samples: self.get_Q_pi_central_nys(
                a_samples, s_samples=s_samples
            )
        else:
            Q_A_fn = lambda a_samples: self.get_Q_pi_central(
                a_samples, s_samples=s_samples
            )
        grad_A = grad(Q_A_fn)(self.K @ s_samples)

        grad_K = (grad_A @ s_samples.T) / n_samples
        true_grad_K = grad(self.get_true_VK)(self.K)

        if self.use_true_grad == True:
            grad_K = grad_K.at[:].set(true_grad_K)
        if self.normalize_grad == True:
            grad_K = grad_K.at[:].set(grad_K / jnp.linalg.norm(grad_K, ord="fro"))

        for i, j in itertools.product(
            range(self.N), range(self.N)
        ):  # elementwise eta tuning
            eta = self.eta
            while True:
                K_new = jnp.empty((self.N, self.N))
                K_new = K_new.at[:].set(self.K)
                K_new = K_new.at[i, j].set(self.K[i, j] - eta * grad_K[i, j])
                if jnp.linalg.norm(self.env.A + self.env.B @ K_new, ord=2) < 1:
                    self.K = self.K.at[i, j].set(K_new[i, j])
                    break
                else:
                    eta /= 2.0
                    if eta < 1e-5:  # just don't update this entry if eta is too small
                        break
                    # print("have to reduce eta, eta = %.3f" %eta)

    # new version of collect data that works for batch of episodes
    def collect_data(self, key=None):
        n_eps = self.data_eps

        # Key for random number generation

        # Run a batch of episodes (for s,a,f,r output is of form T by N by n_eps.)
        start = time.time()
        s_data, a_data, f_s_data, r_data = self.env.run_episode(
            K=self.K, s0=jnp.zeros((self.N, n_eps)), n_eps=n_eps, key=key
        )
        print("running episodes took %.3f seconds" % (time.time() - start))
        start = time.time()
        f_s_data_swapped = jnp.swapaxes(
            f_s_data, 1, 2
        )  # T by n_eps by N, facilitates reshaping later.
        f_s_data_curr = f_s_data_swapped[:-1, :, :].reshape(
            (self.env.T - 1) * n_eps, self.N
        )
        f_s_data_next = f_s_data_swapped[1:, :, :].reshape(
            (self.env.T - 1) * n_eps, self.N
        )
        r_data_swapped = jnp.swapaxes(r_data, 1, 2)
        r_data_next = r_data_swapped[1:, :, :].reshape((self.env.T - 1) * n_eps, self.N)
        print("array swapping took %.3f seconds" % (time.time() - start))

        return (
            s_data,
            a_data,
            f_s_data,
            r_data,
            f_s_data_curr,
            f_s_data_next,
            r_data_next,
        )

    # @jit
    def collect_data_central(self, key=None):
        n_eps = self.data_eps

        # Initialize np arrays for phi, phi_next, and r_next
        phi_np = np.empty((n_eps * (self.env.T - 1), self.m))
        phi_next_np = np.empty((n_eps * (self.env.T - 1), self.m))
        r_next_np = np.empty((n_eps * (self.env.T - 1),))

        # Run a batch of episodes (for s,a,f,r output is of form T by N by n_eps.)
        s_data, a_data, f_s_data, r_data = self.env.run_episode(
            K=self.K, s0=jnp.zeros((self.N, n_eps)), n_eps=n_eps, key=key
        )
        f_s_data_swapped = jnp.swapaxes(
            f_s_data, 1, 2
        )  # T by n_eps by N, facilitates reshaping later.
        f_s_data_curr = f_s_data_swapped[:-1, :, :].reshape(
            (self.env.T - 1) * n_eps, self.N
        )
        f_s_data_next = f_s_data_swapped[1:, :, :].reshape(
            (self.env.T - 1) * n_eps, self.N
        )
        r_data_swapped = jnp.swapaxes(r_data, 1, 2)
        r_data_next = r_data_swapped[1:, :, :].reshape((self.env.T - 1) * n_eps, self.N)

        phi_np = self.get_features_central(f_s_data_curr)
        phi_next_np = self.get_features_central(f_s_data_next)
        r_next_np = jnp.mean(
            r_data_next, axis=1
        )  # average rewards across agents for each sample

        start = time.time()
        phi = jnp.asarray(phi_np)
        phi_next = jnp.asarray(phi_next_np)
        r_next = jnp.asarray(r_next_np)
        print("final changing np to jnp took %.3f seconds" % (time.time() - start))

        return s_data, a_data, f_s_data, r_data, phi, phi_next, r_next

    # output: N by n_samples
    # new version of get_samples for batch data collection
    def get_samples(
        self, s_data=None, buffer_length=10, equal_weights=False, replace=True
    ):
        s_data_swapped = jnp.swapaxes(s_data, 1, 2)
        s_data_concat = s_data_swapped.reshape(self.env.T * self.data_eps, self.N)
        # Uses JAX for operations and random sampling
        key = self.key
        s_samples = jnp.empty(
            (buffer_length, self.N * self.s_dim)
        )  # this works when self.s_dim = 1.
        if equal_weights == True:
            weights = jnp.ones(self.env.T * self.data_eps)
        else:
            # Initialize an empty list to store the values
            array = []
            # Fill the array with the required values
            for t in range(self.env.T):
                array.extend([self.gamma ** (t)] * self.data_eps)
            # Convert the list to a numpy array if needed
            weights = jnp.array(array)
            # print("weights shape", weights.shape)
        weights /= weights.sum()
        # print("a shape", jnp.arange(self.env.T * self.data_eps).shape)
        rand_idx = random.choice(
            self.key,
            jnp.arange(self.env.T * self.data_eps),
            shape=(buffer_length,),
            p=weights,
            replace=replace,
        )
        # print("rand_idx",rand_idx)
        # print("buffer_length", buffer_length)
        s_samples = s_data_concat[rand_idx, :]

        s_samples = s_samples.T  # N by bufffer_length
        # print("s_samples first 100 entries", s_samples[:,:100])
        return s_samples

    def get_approx_V(self, K, s_samples=None):
        # Compute approximate value function V
        pi_samples = jnp.dot(K, s_samples)
        f_samples = self.env.transition(s_samples, pi_samples)
        f_samples = f_samples.T
        approx_V = 0
        # print("f samples shape", f_samples.shape)
        # print("s_samples", s_samples)
        for i in range(self.N):
            phi_fi_samples = self.get_features(f_samples, i)
            # print(" phi fi shape", phi_fi_samples.shape)
            # print("theta i shape", self.theta[i,:].shape)
            # print("this is the dot product (kappa = %d)" %self.kappa, jnp.dot(phi_fi_samples, self.theta[i,:]))
            approx_V += jnp.sum(self.gamma * jnp.dot(phi_fi_samples, self.theta[i, :]))
        r_samples = self.alpha * jnp.sum(s_samples**2, axis=0) + jnp.sum(
            pi_samples**2, axis=0
        )
        approx_V += jnp.sum(r_samples)
        approx_V /= (
            self.N * s_samples.shape[1]
        )  # s_samples.shape[1] is the number of samples
        # print("this is approx V", approx_V)
        # print("this is pi samples", pi_samples)
        return approx_V

    def get_approx_V_central(self, K, s_samples=None):
        # Compute approximate value function V
        pi_samples = jnp.dot(K, s_samples)
        f_samples = self.env.transition(s_samples, pi_samples)
        f_samples = f_samples.T
        if self.use_nys == True:
            phi_f_samples = self.get_features_central_nys(f_samples)
        else:
            phi_f_samples = self.get_features_central(f_samples)
        r_samples = (
            self.alpha * jnp.sum(s_samples**2, axis=0)
            + jnp.sum(pi_samples**2, axis=0)
        ) / self.N
        approx_V = jnp.sum(
            self.gamma * jnp.dot(phi_f_samples, self.theta_central) + r_samples
        )
        approx_V /= s_samples.shape[1]  # s_samples.shape[1] is the number of samples
        return approx_V

    def get_Q_pi(self, a_samples, s_samples=None):
        f_samples = self.env.transition(s_samples, a_samples)
        f_samples = f_samples.T
        approx_Q = 0
        for i in range(self.N):
            phi_fi_samples = self.get_features(f_samples, i)
            approx_Q += jnp.sum(jnp.dot(phi_fi_samples, self.theta[i, :]))
        approx_Q /= self.N
        return approx_Q

    def get_Q_pi_central(self, a_samples, s_samples=None):
        f_samples = self.env.transition(s_samples, a_samples)
        # print("f_samples shape", f_samples.shape)
        f_samples = f_samples.T
        phi_f_samples = self.get_features_central(f_samples)
        r_samples = (
            self.alpha * jnp.sum(s_samples**2, axis=0)
            + jnp.sum(a_samples**2, axis=0)
        ) / self.N
        approx_Q = jnp.sum(
            self.gamma * jnp.dot(phi_f_samples, self.theta_central) + r_samples
        )

        return approx_Q

    def get_Q_pi_central_nys(self, a_samples, s_samples=None):
        f_samples = self.env.transition(s_samples, a_samples)
        f_samples = f_samples.T
        phi_f_samples = self.get_features_central_nys(f_samples)
        r_samples = (
            self.alpha * jnp.sum(s_samples**2, axis=0)
            + jnp.sum(a_samples**2, axis=0)
        ) / self.N

        approx_Q = jnp.sum(
            self.gamma * jnp.dot(phi_f_samples, self.theta_central) + r_samples
        )

        return approx_Q

    def get_Q_pi_central_vec(self, a_samples, s_samples=None):
        f_samples = self.env.transition(s_samples, a_samples)
        f_samples = f_samples.T
        if self.use_nys == True:
            phi_f_samples = self.get_features_central_nys(f_samples)
        else:
            phi_f_samples = self.get_features_central(f_samples)
        r_samples = (
            self.alpha * jnp.sum(s_samples**2, axis=0)
            + jnp.sum(a_samples**2, axis=0)
        ) / self.N
        approx_Q_vec = (
            self.gamma * jnp.dot(phi_f_samples, self.theta_central) + r_samples
        )
        return approx_Q_vec

    # evaluates the cost of a controller K, starting from the fixed point x0 = 0.
    def get_true_VK(self, K, gamma=None):
        if gamma is None:
            gamma = self.gamma
        PK = self.env.find_P_K(K)
        Sigma = self.noise_sig**2 * jnp.eye(self.N)
        VK = jnp.trace(PK @ Sigma) * gamma / (1 - gamma)
        return VK / self.N

    # evaluates the true Q_pi for a controller K
    def get_true_QK(self, K, s_samples, a_samples):
        PK = self.env.find_P_K(K)
        Sigma = self.noise_sig**2 * jnp.eye(self.N)
        VK_zero = jnp.trace(PK @ Sigma) * self.gamma / (1 - self.gamma)
        r_samples = jnp.sum(self.alpha * s_samples**2, axis=0) + jnp.sum(
            a_samples**2, axis=0
        )
        f_s_samples = self.env.A @ s_samples + self.env.B @ a_samples
        QK = (
            r_samples
            + self.gamma * (jnp.diag(f_s_samples.T @ (PK @ f_s_samples)) + VK_zero)
        ) / self.N

        return QK

    # evaluates the true T_pi for a controller K
    def get_true_TK(self, K, s_samples, a_samples):
        PK = self.env.find_P_K(K)
        Sigma = self.noise_sig**2 * jnp.eye(self.N)
        VK_zero = jnp.trace(PK @ Sigma) * self.gamma / (1 - self.gamma)
        f_s_samples = self.env.A @ s_samples + self.env.B @ a_samples
        TK = (jnp.diag(f_s_samples.T @ (PK @ f_s_samples)) + VK_zero) / self.N
        return TK

    # truncates any N by N matrix to only retain non-zero values within a kappa-neighborhood
    def trunc_kappa_controller(self, M):
        mask = jnp.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if ((j - i) % (self.N)) <= self.trunc_kappa:
                    mask = mask.at[i, j].set(1.0)
                    mask = mask.at[j, i].set(1.0)
        M *= mask
        # print("mask", mask)
        return M

    def policy_eval_GD(
        self, f_s, f_s_next, r_next, n_opt_epochs=2000, batchsize=1, alpha=5 * 1e-4
    ):
        def update_theta(i):
            theta_i = jnp.zeros(self.m)
            for j in range(n_opt_epochs):
                f_s_j = f_s[j * batchsize : (j + 1) * batchsize, :]
                f_s_next_j = f_s_next[j * batchsize : (j + 1) * batchsize, :]
                r_next_j = r_next[j * batchsize + 75 : (j + 1) * batchsize + 75, :]
                phi_j = self.get_features(f_s_j, i)
                phi_next_j = self.get_features(f_s_next_j, i)

                td_errors = jnp.dot(phi_j, theta_i) - (
                    r_next_j[:, i] + self.gamma * jnp.dot(phi_next_j, theta_i)
                )

                w_update = jnp.dot(td_errors, phi_j)

                theta_i = theta_i.at[:].set(theta_i - alpha * w_update / batchsize)
                if j % 5 == 0:
                    print(
                        "average td errors before update, i = %d, j = %d" % (i, j),
                        jnp.mean(td_errors**2),
                    )
                    td_errors_after = (
                        r_next_j[:, i]
                        + self.gamma * jnp.dot(phi_next_j, theta_i)
                        - jnp.dot(phi_j, theta_i)
                    )
                    print(
                        "average td errors after update, i = %d, j = %d" % (i, j),
                        jnp.mean(td_errors_after**2),
                    )
            return i, theta_i

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(update_theta, range(self.N)))

        for i, theta_i in results:
            self.theta = self.theta.at[i, :].set(theta_i)

    def policy_eval_GD_central(
        self, f_s, f_s_next, r_next, n_opt_epochs=10, batchsize=5000, alpha=1 * 1e-3
    ):
        theta = jnp.zeros(self.m)
        optimizer = optax.adam(alpha)
        opt_state = optimizer.init(theta)
        for j in range(n_opt_epochs):
            # try always using same data
            f_s_j = f_s[0 * batchsize : batchsize, :]
            f_s_next_j = f_s_next[0 * batchsize : batchsize, :]
            r_next_j = r_next[0 * batchsize : batchsize, :]
            phi_j = self.get_features_central(f_s_j)
            phi_next_j = self.get_features_central(f_s_next_j)

            print("phi j shape", phi_j.shape)

            td_errors = jnp.dot(phi_j, theta) - (
                jnp.sum(r_next_j, axis=1) / self.N
                + self.gamma * jnp.dot(phi_next_j, theta)
            )

            LHS = jnp.dot(phi_j.T, phi_j) - self.gamma * jnp.dot(phi_j.T, phi_next_j)
            RHS = jnp.dot(phi_j.T, (jnp.sum(r_next_j, axis=1) / self.N))
            print("LHS shape", LHS.shape)
            print("RHS shape", RHS.shape)
            if j % 100 == 0:
                print(
                    "mean ||Ax-b||^2 before update, j = %d" % j,
                    jnp.mean((LHS @ theta - RHS) ** 2),
                )

            w_update = jnp.dot(td_errors, phi_j)
            w_update_final = jnp.dot(
                (jnp.dot(phi_j.T, phi_j) - self.gamma * jnp.dot(phi_j.T, phi_next_j)),
                w_update,
            )

            theta = theta.at[:].set(theta - alpha * w_update_final / (batchsize**2))
            td_errors_after = (
                jnp.sum(r_next_j, axis=1) / self.N
                + self.gamma * jnp.dot(phi_next_j, theta)
                - jnp.dot(phi_j, theta)
            )

            if j % 100 == 0:
                print(
                    "mean ||Ax-b||^2 after update, j = %d" % j,
                    jnp.mean((LHS @ theta - RHS) ** 2),
                )

        self.theta_central = self.theta_central.at[:].set(theta)

    def mse_loss(self, params, f_s, f_s_next, r_next):
        pred = forward(params, self.fourier_w, self.fourier_b, f_s)
        target = r_next + self.gamma * jax.lax.stop_gradient(
            forward(params, self.fourier_w, self.fourier_b, f_s_next)
        )
        # print("sq err", (pred - target)**2)
        return jnp.mean((pred - target) ** 2)

    def policy_eval_rf(self, f_s, f_s_next, r_next, n_epochs=10):
        for i in range(n_epochs):
            grads = jax.grad(self.mse_loss)(self.rf_params, f_s, f_s_next, r_next)

            updates, self.opt_state = self.rf_optimizer.update(grads, self.opt_state)
            self.rf_params = optax.apply_updates(self.rf_params, updates)

    def run(self, eval_freq=1, n_eval_eps=1, n_data_eps=10):
        E = self.n_epochs
        costs = np.empty(E)
        K_diff = np.empty(E)
        date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(
            f"runs/N={self.N}/kappa={self.kappa}/trunc_kappa={self.trunc_kappa}/data_eps_size={self.data_eps}/seed={self.key_int}/lr={args.lr}/jax_experiment_rf_{date_time_str}_rf_num={self.m}"
        )
        for i in range(E):
            start_i = time.time()
            if i % eval_freq == 0:
                print("true cost of current controller", self.get_true_VK(self.K))
                print(
                    "A+BK spectral norm",
                    jnp.linalg.norm(self.env.A + self.env.B @ self.K, ord=2),
                )
                opt_K = self.env.find_opt_K()
                higher_gamma = 0.95
                opt_K_gamma = self.env.find_opt_K(gamma=higher_gamma)

                trunc_opt_K = self.trunc_kappa_controller(opt_K)

                K_diff[i] = jnp.linalg.norm(trunc_opt_K - self.K)
                print("cost of trunc opt K", self.get_true_VK(trunc_opt_K))
                print("cost of opt K", self.get_true_VK(opt_K))
                print(
                    "cost of opt K, higher gamma = %.2f" % higher_gamma,
                    self.get_true_VK(opt_K_gamma, gamma=higher_gamma),
                )
                opt_cost = self.get_true_VK(opt_K)
                print("true optimal cost is", self.env.find_opt())

            # Collect data from environment
            start = time.time()
            (
                s_data,
                a_data,
                f_s_data,
                r_data,
                f_s_data_curr,
                f_s_data_next,
                r_data_next,
            ) = self.collect_data(key=self.keys[i])
            print("i", i)
            self.s_data = s_data
            print(
                "data collected, epoch: %d. this took %.3f seconds"
                % (i, time.time() - start)
            )

            # try new policyy eval
            start = time.time()
            self.policy_eval(f_s_data_curr, f_s_data_next, r_data_next)
            print("new policy eval took %.1f seconds" % (time.time() - start))

            # Evaluate policy with collected data
            start_eval = time.time()
            approx_V_s0_zero = self.get_approx_V(
                K=self.K, s_samples=jnp.zeros((self.N, 1))
            )
            print(
                "(epoch %d) approx cost starting at s0 = 0" % i,
                approx_V_s0_zero * self.gamma,
            )  # the times self.gamma is to account for the fact that the features try to learn T^pi, but we want to approx V^pi
            current_cost = self.get_true_VK(self.K)
            print(
                "(epoch %d) true cost of current controller" % i,
                self.get_true_VK(self.K),
            )
            # print("current K", self.K)
            print("evaluation took %.3f seconds" % (time.time() - start_eval))

            n_samples = 100

            # randomly select some samples from data

            rand_t = jax.random.choice(
                key=self.keys[i],
                a=jnp.arange(self.env.T),
                shape=(n_samples,),
                replace=True,
            )
            rand_eps = jax.random.choice(
                key=self.keys[i],
                a=jnp.arange(n_data_eps),
                shape=(n_samples,),
                replace=True,
            )
            s_samples = jnp.empty((n_samples, self.N))
            a_samples = jnp.empty((n_samples, self.N))
            for n in range(n_samples):
                s_samples = s_samples.at[n, :].set(
                    self.s_data[rand_t[n], :, rand_eps[n]]
                )
                a_samples = a_samples.at[n, :].set(jnp.dot(self.K, s_samples[n, :]))

            print("self K shape", self.K.shape)
            true_QK = self.get_true_QK(self.K, s_samples.T, a_samples.T)
            approx_QK = self.get_Q_pi_central_vec(
                a_samples=a_samples.T, s_samples=s_samples.T
            )

            writer.add_scalar(
                "Policy evaluation error", jnp.mean(true_QK - approx_QK) ** 2, i
            )

            costs[i] = self.get_true_VK(self.K)
            writer.add_scalar("Cost", costs[i], i)

            # Update policy using Gradient Descent
            start = time.time()

            self.policy_update()
            print("policy update took %.3f seconds" % (time.time() - start))

            print(
                "in total, this iteration took %.3f seconds" % (time.time() - start_i)
            )

            # update epoch
            self.e += 1
        plt.plot(
            np.arange(E),
            (costs - opt_cost) / opt_cost,
            label="truncated SDEC, kappa = %d" % self.kappa,
        )
        plt.plot(
            np.arange(E),
            np.ones(E) * (self.get_true_VK(trunc_opt_K) - opt_cost) / opt_cost,
            label="truncation of optimal controller, kappa = %d" % self.kappa,
            linestyle="dashdot",
        )
        plt.title("N = %d" % self.N)
        plt.xlabel("epochs")
        plt.ylabel("average (over N) cost")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.savefig(
            "figures/hvac/learning_trajectory_N=%d_kappa=%d_trunc_kappa=%d_m=%d_n_data_eps=%d_key=%d_batch_rf.pdf"
            % (
                self.N,
                self.kappa,
                self.trunc_kappa,
                self.m,
                self.data_eps,
                self.key_int,
            )
        )
        plt.close()

        print(
            "kappa=%d: trunc opt cost to be saved" % self.kappa,
            self.get_true_VK(trunc_opt_K),
        )
        outfile = (
            "learning_trajectory_npz/hvac/N=%d_kappa=%d_trunc_kappa=%d_m=%d_n_data_eps=%d_key=%d_nepochs=%d_batch_rf.npz"
            % (
                self.N,
                self.kappa,
                self.trunc_kappa,
                self.m,
                self.data_eps,
                self.key_int,
                self.n_epochs,
            )
        )
        np.savez(
            outfile,
            costs=costs,
            trunc_opt_cost=self.get_true_VK(trunc_opt_K),
            opt_cost=self.get_true_VK(opt_K),
        )

        plt.plot(np.arange(E), K_diff, label="truncated SDEC, kappa = %d" % self.kappa)
        plt.title("N = %d" % self.N)
        plt.xlabel("epochs")
        plt.ylabel("difference between trunc_opt_K and current K")
        plt.grid()
        plt.legend()
        plt.savefig(
            "figures/hvac/learning_trajectory_N=%d_kappa=%d_trunc_kappa=%d_m=%d_n_data_eps=%d_key=%d_batch_Kdiff.pdf"
            % (
                self.N,
                self.kappa,
                self.trunc_kappa,
                self.m,
                self.data_eps,
                self.key_int,
            )
        )
        plt.close()

    def run_central(self, eval_freq=1, n_eval_eps=1, n_data_eps=10):
        E = self.n_epochs
        costs = np.empty(E)
        date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(
            f"runs/N={self.N}/data_eps_size={self.data_eps}/jax_experiment_rf_{date_time_str}_rf_num={self.m}"
        )
        for i in range(E):
            start_i = time.time()
            if i % eval_freq == 0:
                # Print evaluation cost
                print("true cost of current controller", self.get_true_VK(self.K))
                opt_K = self.env.find_opt_K()

                print("cost of opt K", self.get_true_VK(opt_K))
                opt_cost = self.get_true_VK(opt_K)
                print("true optimal cost is", self.env.find_opt())

            # update Nystrom features every time the get_nys_samples() function will use the current K
            if self.use_nys == True:
                self.nys_samples = self.get_nys_samples()
                self.gen_nystrom_central()

            # Collect data from environment
            start = time.time()
            (
                s_data,
                a_data,
                f_s_data,
                r_data,
                phi,
                phi_next,
                r_next,
            ) = self.collect_data_central(key=self.keys[i])
            print("keys i", self.keys[i])
            self.s_data = s_data
            print("data collection took %.3f seconds" % (time.time() - start))

            # Evaluate policy with collected data
            start = time.time()

            self.policy_eval_central(phi, phi_next, r_next)
            print("policy eval took %.3f seconds" % (time.time() - start))
            print(
                "(epoch %d) true cost of current controller" % i,
                self.get_true_VK(self.K),
            )

            # check if approx Q^pi is similar to true Q^pi
            n_samples = 100

            # randomly select some samples from data

            rand_t = jax.random.choice(
                key=self.keys[i],
                a=jnp.arange(self.env.T),
                shape=(n_samples,),
                replace=True,
            )
            rand_eps = jax.random.choice(
                key=self.keys[i],
                a=jnp.arange(n_data_eps),
                shape=(n_samples,),
                replace=True,
            )
            s_samples = jnp.empty((n_samples, self.N))
            a_samples = jnp.empty((n_samples, self.N))
            for n in range(n_samples):
                s_samples = s_samples.at[n, :].set(
                    self.s_data[rand_t[n], :, rand_eps[n]]
                )
                a_samples = a_samples.at[n, :].set(jnp.dot(self.K, s_samples[n, :]))

            # print("self K shape", self.K.shape)
            true_QK = self.get_true_QK(self.K, s_samples.T, a_samples.T)
            approx_QK = self.get_Q_pi_central_vec(
                a_samples=a_samples.T, s_samples=s_samples.T
            )

            writer.add_scalar(
                "Policy evaluation error", jnp.mean(true_QK - approx_QK) ** 2, i
            )

            costs[i] = self.get_true_VK(self.K)
            writer.add_scalar("Cost", costs[i], i)

            # Update policy using Gradient Descent
            start = time.time()
            self.policy_update_central()
            print("policy update took %.3f seconds" % (time.time() - start))

            print("iteration %d took %.3f seconds" % (i, time.time() - start_i))

        # update epoch
        # self.e += 1
        plt.plot(np.arange(E), (costs - opt_cost) / opt_cost, label="centralized SDEC")
        plt.title("N = %d" % self.N)
        plt.xlabel("epochs")
        plt.ylabel("average (over N) cost")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.savefig(
            "figures/hvac/new/rf_learning_trajectory_N=%d_m=%d_n_data_eps=%d_key=%d_batch_centralized.pdf"
            % (self.N, self.m, self.data_eps, self.key_int)
        )
        plt.close()
        writer.close()


# See Yingying's paper for details
class hvac:
    def __init__(self, N=10, gamma=0.9, T=10, v=200, zeta=1.0, alpha=3, delta=20):
        self.N = N
        self.gamma = gamma
        self.T = T
        self.v = v * jnp.ones(N)
        self.zeta = zeta
        self.delta = delta
        self.alpha = alpha
        self.noise_sig = 1.0
        self.A = self.gen_banded_A(N)
        self.B = self.gen_banded_B(N)

    def gen_banded_B(self, N, off_diag_scale=0.7):
        B = jnp.diag(10 * self.delta / self.v)
        for i in range(N):
            B = B.at[i, (i + 1) % N].set(off_diag_scale * 10 * self.delta / self.v[i])
            B = B.at[i, (i - 1) % N].set(off_diag_scale * 10 * self.delta / self.v[i])
        print("B", B)
        return B

    def gen_banded_A(self, N):
        A = jnp.diag(1 - (2 * self.delta) / (self.v * self.zeta))
        for i in range(N):
            A = A.at[i, (i + 1) % N].set(self.delta / (self.v[i] * self.zeta))
            A = A.at[i, (i - 1) % N].set(self.delta / (self.v[i] * self.zeta))
        if N == 1:  # for testing only
            A = A.at[0, 0].set(0.8)
        return A

    def transition(self, s, a):
        return self.A @ s + self.B @ a

    # 1-step transition with noise
    def transition_noisy(self, s, a, n_eps=30, key=None):
        s_new = (
            self.A @ s
            + self.B @ a
            + self.noise_sig * jax.random.normal(key, shape=(self.N, n_eps))
        )
        return s_new

    # new version of run episode that works for batch of starting points
    def run_episode(self, K, n_eps=30, s0=None, key=None):
        s_all = jnp.empty((self.T, self.N, n_eps))
        a_all = jnp.empty((self.T, self.N, n_eps))
        r_all = jnp.empty((self.T, self.N, n_eps))
        f_s_all = jnp.empty((self.T, self.N, n_eps))
        s = s0 if s0 is not None else jnp.zeros(self.N, n_eps)
        keys = random.split(key, self.T)
        for i in range(self.T):
            a = K @ s
            r = self.alpha * s**2 + a**2
            f_s = self.transition(s, a)
            s_all = s_all.at[i, :, :].set(s)
            a_all = a_all.at[i, :, :].set(a)
            r_all = r_all.at[i, :, :].set(r)
            f_s_all = f_s_all.at[i, :, :].set(f_s)
            s = self.transition_noisy(
                s, a, n_eps=n_eps, key=keys[i]
            )  # assuming transition incorporates noise
        return (s_all, a_all, f_s_all, r_all)

    # this finds the P_Kstar for value function of optimal action sequence
    # for the given enviroment
    def find_P(self, A, B, Q, R, gamma, tot_iters=50):
        P = Q
        print(B.shape, P.shape, A.shape)
        for i in jnp.arange(tot_iters):
            BtPA = B.T @ (P @ A)
            Rplus_inv = jnp.linalg.inv(R + gamma * jnp.transpose(B) @ (P @ B))
            AtPA = jnp.transpose(A) @ (P @ A)
            P = Q - gamma**2 * jnp.transpose(BtPA) @ (Rplus_inv @ BtPA) + gamma * AtPA
        return P

    # find P for specific K
    def find_P_K(self, K, tot_iters=50):
        A = self.A
        B = self.B
        Q = self.alpha * jnp.eye(self.N)
        R = jnp.eye(self.N)
        gamma = self.gamma
        P = Q
        for i in jnp.arange(tot_iters):
            P = Q + K.T @ (R @ K) + gamma * (A + B @ K).T @ (P @ (A + B @ K))
        return P

    # returns optimal value
    def find_opt(self):
        Popt = self.find_P(
            self.A, self.B, self.alpha * jnp.eye(self.N), jnp.eye(self.N), self.gamma
        )
        Sigma = self.noise_sig**2 * jnp.eye(self.N)
        return jnp.trace(Popt @ Sigma) * self.gamma / (1 - self.gamma)

    def find_opt_K(self, gamma=None):
        if gamma is None:
            gamma = self.gamma
        R = jnp.eye(self.N)
        Q = self.alpha * jnp.eye(self.N)
        P = self.find_P(self.A, self.B, Q, R, gamma)
        K = (
            gamma
            * jnp.linalg.inv(R + gamma * self.B.T @ (P @ self.B))
            @ (self.B.T @ (P @ self.A))
        )
        return -K


# create N masks such that multipling any matrix (whose columns index the agents) by mask i will only retain columns within a kappa-neighborhood of i
def trunc_kappa_masks(trunc_kappa=0, N=None):
    masks = jnp.zeros((N, N, N))
    for i in range(N):
        mask = jnp.zeros((N, N))
        for j in range(N):
            if ((j - i) % (N)) <= trunc_kappa:
                mask = mask.at[j, j].set(1.0)
        masks = masks.at[i, :, :].set(mask)
    return masks


def get_features(i, indices, f_s, omega, b):
    f_s_i = f_s[:, indices]
    phi_f_s_i = jnp.cos(jnp.dot(f_s_i, omega[i, :, :]) + b[i, :])
    return phi_f_s_i


def main(args):
    N = 50

    gamma = 0.75

    T = 20
    alpha = 3
    delta = 20
    env = hvac(N=N, gamma=gamma, T=T, alpha=alpha, delta=delta)
    K0 = jnp.zeros((N, N))

    n_epochs = 40

    trunc_kappas = [0, 1, 2, 3]

    kappas = [0, 1, 2, 2]

    m_vec = [30, 200, 800, 800]

    n_data_eps_vec = [100, 200, 500, 1000]

    n_seeds = 5
    outfiles = [[None] * n_seeds for _ in range(len(m_vec))]

    run_central = False
    if run_central == True:
        m = 800
        n_data_eps = 1000
        alg = algorithm(
            env=env,
            N=N,
            kappa=0,
            trunc_kappa=trunc_kappa,
            gamma=gamma,
            K0=K0,
            m=m,
            n_epochs=n_epochs,
            eta=2 * 1e-2,
            use_true_grad=False,
            equal_weights=True,
            normalize_grad=True,
            use_nys=False,
            data_eps=n_data_eps,
            key_int=key,
            alpha=alpha,
            noise_sig=1.0,
        )
        alg.run_central(n_eval_eps=1, n_data_eps=n_data_eps)
    else:
        for i in range(len(kappas)):
            kappa = kappas[i]
            print("KAPPA", kappa)
            trunc_kappa = trunc_kappas[i]
            print("trunc KAPPA", trunc_kappa)

            m = m_vec[i]
            n_data_eps = n_data_eps_vec[i]

            for num in range(n_seeds):
                key = args.seed + num
                alg = algorithm(
                    env=env,
                    N=N,
                    kappa=kappa,
                    trunc_kappa=trunc_kappa,
                    gamma=gamma,
                    K0=K0,
                    m=m,
                    n_epochs=n_epochs,
                    eta=args.lr,
                    use_true_grad=False,
                    equal_weights=True,
                    normalize_grad=True,
                    use_nys=False,
                    data_eps=n_data_eps,
                    key_int=key,
                    alpha=alpha,
                )

                alg.run(n_eval_eps=1)

                outfiles[i][num] = (
                    "learning_trajectory_npz/hvac/N=%d_kappa=%d_trunc_kappa=%d_m=%d_n_data_eps=%d_key=%d_nepochs=%d_batch_rf.npz"
                    % (N, kappa, trunc_kappa, m, n_data_eps, key, n_epochs)
                )

        # plot the information of different kappas on same plot
        for i in range(len(kappas)):
            kappa = kappas[i]
            trunc_kappa = trunc_kappas[i]
            costs = [None] * n_seeds
            for num in range(n_seeds):
                data = np.load(outfiles[i][num])
                costs[num] = data["costs"]
                trunc_opt_cost = data["trunc_opt_cost"]
                opt_cost = data["opt_cost"]
            stacked_costs = np.vstack(costs)
            # Compute the mean and standard deviation across the rows (axis=0)
            mean_costs = np.mean(stacked_costs, axis=0)
            std_costs = np.std(stacked_costs, axis=0)

            # Define the upper and lower bounds for the confidence interval (1 std)
            upper_bound = mean_costs + std_costs
            lower_bound = mean_costs - std_costs

            # Create the plot
            x = np.arange(len(mean_costs))  # x-axis (index of elements)

            line = plt.plot(
                np.arange(n_epochs),
                mean_costs,
                label=r"Truncated Spectral-Rep, $\kappa_\pi$ = %d" % (trunc_kappa),
            )[0]
            plt.fill_between(
                np.arange(n_epochs),
                lower_bound,
                upper_bound,
                color=line.get_color(),
                alpha=0.5,
            )

        plt.plot(
            np.arange(n_epochs),
            np.ones(n_epochs) * trunc_opt_cost,
            label=r"optimal controller",
            linestyle="dashdot",
        )

        plt.title("N = %d" % N)
        plt.xlabel("epochs")
        plt.ylabel(r"$J(K)$")
        plt.grid()
        plt.yscale("log")
        plt.legend(loc="upper right")

        dir_path = "figures/new/hvac"
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(
            dir_path
            + "/learning_trajectory_N=%d_trunc_kappa=%d_kappa_comparison_m=%d_n_data_eps=%d_key=%d_nepochs=%d_lr=%d_batch_rf.pdf"
            % (N, trunc_kappa, m, n_data_eps, key, n_epochs, args.lr),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Step 2: Add arguments
    parser.add_argument("--seed", type=int, default=4, help="Random seed (default: 42)")
    parser.add_argument(
        "--lr", type=float, default=0.2, help="Learning rate (default: 0.001)"
    )

    # Step 3: Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)

    # main()
