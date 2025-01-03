import jax.numpy as jnp

def edm_ema_scales_schedules(step, config, steps_per_epoch):
  # ema_halflife_kimg = 500  # from edm


  ema_halflife_kimg = config.get('ema_halflife_kimg',100000)  # log(0.5) / log(0.999999) * 128 / 1000 = 88722 kimg, from flow
  ema_halflife_nimg = ema_halflife_kimg * 1000

  # ema_rampup_ratio = 0.05
  ema_rampup_ratio = config.get('ema_rampup_ratio',0.135)
  bz = config.get('batch_size', 2048)
  ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * bz * ema_rampup_ratio)

  ema_beta = 0.5 ** (bz / jnp.maximum(ema_halflife_nimg, 1e-8))

  # ema_beta = jnp.ones((), dtype=jnp.float32) * 0.999 # don't tune ema for now
  
  scales = jnp.ones((1,), dtype=jnp.int32)
  return ema_beta, scales

import matplotlib.pyplot as plt
xs = jnp.arange(0, 100000, 1000)
cfgs = [{
  'ema_halflife_kimg': 100000,
    'ema_rampup_ratio': 0.135,
}, {
    'ema_halflife_kimg': 1145,
        'ema_rampup_ratio': 0.08,
}
]
for cfg in cfgs:
    ys = [edm_ema_scales_schedules(x, cfg | {'batch_size': 2048}, None)[0] for x in xs]
    plt.plot(xs, jnp.log(1-jnp.array(ys)), label=f'king={cfg["ema_halflife_kimg"]}, ramp={cfg["ema_rampup_ratio"]}')
plt.legend()
plt.savefig('ema_vis.png')