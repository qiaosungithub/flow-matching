import jax
import jax.numpy as jnp
from jax.scipy.special import erf

t_min = 0.002 
t_max = 80.0
rho = 7
# P_mean = -0.5
P_mean = -1.1
# P_std = 0.6
P_std = 2.0

def compute_t(indices, scales):
  """
  indices \in [1, scales]
  
  from big noise to small noise, which is different with Song's code
  那其实就要不一样，不高兴的
  """
  # t_max = 80
  # t_min = 0.002
  # rho = 7.0
  # t = self.t_max ** (1 / self.rho) + indices / (scales - 1) * (
  #     self.t_min ** (1 / self.rho) - self.t_max ** (1 / self.rho)
  # )
  t = t_min ** (1 / rho) + (indices - 1) / (scales - 1) * (
      t_max ** (1 / rho) - t_min ** (1 / rho)
  )
  t = t**rho
  return t

def sample_icm_t(
  samples_shape,
  # model,
  scale: int,
  rng,
) -> jnp.array:
  """
  这个应该返回 [1, scales-1] 之内的数
  
  from improved CM. util function
  ----------
  Draws timesteps from a lognormal distribution.

  Parameters
  ----------
  samples_shape: the shape of the samples to draw
  model
  scale: a scalar
  rng: a key

  Returns
  -------
  Tensor
      Timesteps drawn from the lognormal distribution.

  References
  ----------
  [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
  """
  # first compute sigma
  indices = jnp.arange(scale) + 1 # [1, scale]
  sigmas = compute_t(indices, scale)

  mean = P_mean
  std = P_std

  # pdf = erf((jnp.log(sigmas[:-1]) - mean) / (std * jnp.sqrt(2))) - erf(
  #       (jnp.log(sigmas[1:]) - mean) / (std * jnp.sqrt(2))
  #   )
  pdf = erf((jnp.log(sigmas[1:]) - mean) / (std * jnp.sqrt(2))) - erf(
        (jnp.log(sigmas[:-1]) - mean) / (std * jnp.sqrt(2))
    )
  assert pdf.ndim == 1, 'pdf should be 1D'
  # pdf = pdf / jnp.sum(pdf) # 多余的，没准反而高兴

  # print(f"mean: {mean}")
  # print(f"std: {std}")

  # print(f"sigmas: {sigmas}")
  # print(f"pdf: {pdf}")

  timesteps = jax.random.choice(rng, a=jnp.arange(1,scale), shape=samples_shape) # i is chosen from [1, scale-1]
  # timesteps = jax.random.choice(rng, a=jnp.arange(1,scale), shape=samples_shape, p=pdf) # i is chosen from [1, scale-1]

  # print(f"timesteps: {timesteps}")

  return timesteps

rng = jax.random.PRNGKey(0)
timesteps = sample_icm_t((10000,), 1280, rng)
sigmas = compute_t(timesteps, 1280)

sequential_sigmas = compute_t(jnp.arange(1, 10), 10)


x = jax.device_get(sigmas)
# plot as histogram
import matplotlib.pyplot as plt
import numpy as np
# plt.hist(np.log(x), bins=100)
# plot frequency, instead of counts
# plt.hist(np.log10(x), bins=100, density=True)
plt.hist(np.log(x), bins=100, density=True)

# plot straight lines at x=log(0.002) and x=log(80)
# plt.axvline(np.log(t_min), color='r', linestyle='dashed', linewidth=2)
# plt.axvline(np.log(t_max), color='r', linestyle='dashed', linewidth=2)

# plot the sequential sigmas
plt.plot(np.log(sequential_sigmas), np.zeros_like(sequential_sigmas), 'ro')

plt.savefig("timesteps.png")