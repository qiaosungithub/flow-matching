import jax.numpy as jnp
import jax

def langevin_step(x, state, rngs, config):
    """
    This is a not pmap version
    ---
    input:
        x: init value (b, h, w, c)
        state: train state
        rngs
        config: evalu config
    output:
        x_new: new value
        mean_of_t: mean of energy
        grad_norm: grad norm
        signal_to_noise_ratio: grad_norm * step_lr / eps
    ---
    x_new = x - step_lr * gradient + noise * (eps)
    it should be that step_lr = eps**2/2
    """
    # get gradient
    def loss_fn(input):
        outputs = state.apply_fn(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state, True, input, jnp.zeros_like(x), jnp.zeros((x.shape[0])))
        t_pred, new_batch_stats, new_rng_states = outputs
        loss = jnp.sum(t_pred)

        return loss, (new_batch_stats, new_rng_states, t_pred) # for debug
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(x)
    grad_norm = (jnp.mean(grad**2))**0.5
    sum_of_t = aux[0]
    assert grad.shape == x.shape
    x = x - config.step_lr * grad + config.eps * jax.random.normal(rngs.sample(), x.shape)
    signal_to_noise_ratio = grad_norm * config.step_lr / config.eps
    return x, sum_of_t/x.shape[0], grad_norm, signal_to_noise_ratio