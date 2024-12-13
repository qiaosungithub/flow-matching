import jax
import jax.numpy as jnp

def edm_ema_scales_schedules(step, config, steps_per_epoch):
    ema_halflife_kimg = 2000  # from edm
    # ema_halflife_kimg = 50000  # log(0.5) / log(0.999999) * 128 / 1000 = 88722 kimg, from flow
    ema_halflife_nimg = ema_halflife_kimg * 1000

    # ema_rampup_ratio = 0.2
    ema_rampup_ratio = 0.05
    ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * config.batch_size * ema_rampup_ratio)

    ema_beta = 0.5 ** (config.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
    scales = jnp.ones((1,), dtype=jnp.int32)
    return ema_beta, scales

# step: 0 -> 48000
# config.batch_size: 2048
config = lambda: None
config.batch_size = 2048

# plot scales
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 48000, 100)
y = np.array([edm_ema_scales_schedules(step, config, None)[0] for step in x])
# plt.plot(x, y)
plt.plot(x, np.log(1-y))
plt.plot(x, np.log(jnp.array(1-0.9993)).repeat(len(x)))
plt.savefig('ema.png')