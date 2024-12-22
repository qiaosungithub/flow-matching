import jax
import jax.numpy as jnp

def custom_scale_schedule(d):
    # d is a dict:
    # {10: 1/4, 20: 1/2, 30: 1/4}
    # each key is a "scale" value, and each value is how much ratio of steps we spend on the scale
    cumsum = jnp.cumsum(jnp.array(list(d.values())))
    assert 0.9999 < cumsum[-1] < 1.0001, f"cumsum[-1] should be 1, but got {cumsum[-1]}"
    scales = jnp.array(list(d.keys()))
    def scale_fn(step):
        idx = jnp.argmax(cumsum > step)
        return scales[idx] + 1
    return scale_fn
  
custom_scales_fn_1 = custom_scale_schedule({
    10: 3/16,
    20: 3/16,
    40: 3/16,
    80: 3/16, 
    160: 1/16,
    320: 1/16,
    640: 1/16,
    1280: 1/16,
})

total_steps = 48000
x = jnp.arange(0, total_steps, 10)
y = jnp.array([custom_scales_fn_1(step / total_steps) for step in x])

import matplotlib.pyplot as plt 
plt.plot(x, y)
plt.savefig('custom_scales_fn_1.png')