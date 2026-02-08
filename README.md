# Sequential Monte Carlo (SMC) Tempering Sampler

This repository implements a **Sequential Monte Carlo tempering sampler** with **Metropolis–Hastings (MH) rejuvenation** for Bayesian parameter inference. The sampler targets a sequence of intermediate distributions that smoothly interpolate between the prior and posterior, using adaptive tempering based on (conditional) effective sample size. 

Main entry points (in `SMC.py`): 
- `SMCTempering`: the SMC tempering algorithm (adaptive tempering, reweighting, resampling, MH rejuvenation, adaptive random-walk proposals). 
- `BPE_SMC`: a convenience wrapper that wires priors, scaling, and the target data, then runs `SMCTempering`. 

---

## Core algorithm

### Scaling and optional logit reparameterisation

Before sampling, it helps to put parameters and outputs on "friendly" scales so the algorithm can take sensible step sizes.

**Input scaling (parameters).** Each parameter \(x_k\) comes with an allowed range \([l_k, u_k]\). We rescale it to a number between 0 and 1:

\[
\theta_k = \frac{x_k-l_k}{u_k-l_k}\in[0,1],\qquad
x_k = l_k + \theta_k\,(u_k-l_k).
\]

So the sampler works with \(\theta\) (always between 0 and 1), and we convert back to \(x\) whenever we need to run the simulator.

**Output scaling (model outputs).** If different outputs have very different sizes, we divide each output by a chosen "typical size" \(s_j>0\):

\[
\tilde y_j = \frac{y_j}{s_j},\qquad y_j = \tilde y_j\,s_j.
\]

Likelihood calculations are then done using the scaled outputs \(\tilde y\). It makes sense to choose \(s\) as the observed target data so each output is considered equally.

**Scaled forward model.** If the original simulator is \(f\) (it takes \(x\) and returns \(y\)), then what the sampler effectively sees is:

\[
\tilde f(\theta) = \frac{f(x)}{s},\qquad x = l + \theta\,(u-l),
\]

i.e., "unscale inputs → run simulator → scale outputs".

**Optional logit reparameterisation.** A further change of variables maps \(\theta\in(0,1)^d\) to an unconstrained state \(z\in\mathbb R^d\) componentwise via

\[
z_k = \mathrm{logit}(\theta_k)=\log\frac{\theta_k}{1-\theta_k},\qquad
\theta_k = \sigma(z_k)=\frac{1}{1+e^{-z_k}}.
\]

Under this transform, densities in \(z\)-space include the Jacobian. For example, if the prior is uniform in \(\theta\) over \([0,1]^d\), then up to an additive constant,

\[
\log p(z)=\sum_{k=1}^d \log\!\big(\sigma(z_k)\,[1-\sigma(z_k)]\big) + \mathrm{const}.
\]

---

Let `p(x)` be the prior density and `L(x)` the likelihood for parameters `x ∈ R^d`. The sampler constructs a tempering path:

### Tempering
A sequence of distributions

\[
\pi_{\lambda}(x) \propto p(x)\,L(x)^{\lambda}, \qquad 0=\lambda_0 < \lambda_1 < \dots < \lambda_T=1,
\]

so that \(\pi_{\lambda_0}\) is the prior and \(\pi_{\lambda_T}\) is the posterior. 

In the implementation, \(\Delta\lambda_t=\lambda_{t+1}-\lambda_t\) is chosen adaptively using a **conditional ESS** criterion. 

### Reweighting
Given particles \(\{x_i^{(t)}\}_{i=1}^N\) and weights \(w_i^{(t)}\) at \(\lambda_t\), the incremental weight update for \(\lambda_{t+1}=\lambda_t+\Delta\lambda_t\) is:

\[
\tilde w_i^{(t+1)} = w_i^{(t)}\,\exp\!\left(\Delta\lambda_t\,\log L(x_i^{(t)})\right)
= w_i^{(t)}\,L(x_i^{(t)})^{\Delta\lambda_t},
\]

then normalize \(w_i^{(t+1)}=\tilde w_i^{(t+1)}/\sum_j \tilde w_j^{(t+1)}\). 

### Resampling
When the effective sample size

\[
\mathrm{ESS}(w)=\frac{1}{\sum_{i=1}^N (w_i)^2}
\]

falls below a threshold, resample indices \(a_i \sim \text{Categorical}(w)\) and set \(x_i \leftarrow x_{a_i}\) with weights reset to \(1/N\). The code uses **systematic resampling**. 

### Rejuvenation
After resampling, apply an MH kernel that leaves \(\pi_{\lambda_{t+1}}\) invariant:

\[
x' = x + \epsilon,\quad \epsilon \sim \mathcal N(0,\Sigma_{\text{prop}}),
\]

\[
\alpha(x,x') = \min\!\left(1, \exp\left[
\lambda_{t+1}(\log L(x')-\log L(x)) + (\log p(x')-\log p(x))
\right]\right).
\]

This MH step is repeated `mh_steps` times at each resampling event. 

### How proposals are calculated
The sampler uses a random-walk Gaussian proposal with covariance

\[
\Sigma_{\text{prop}} = \left(\frac{2.38^2}{d}\right)\,s^2\,\widehat\Sigma + \varepsilon I,
\]

where \(\widehat\Sigma\) is the (optionally weight-adjusted) empirical covariance of the current particle cloud, \(d\) is dimension, \(s\) is a scalar proposal scale, and \(\varepsilon I\) is a small diagonal stabilizer. 

### Robbins–Monro update (proposal scale)
After each MH step, the scalar proposal scale \(s\) is adapted via a stochastic approximation update on \(\log s\):

\[
\log s \leftarrow \log s + \eta\,(\hat a - a^\star),
\]

where \(\hat a\) is the observed acceptance rate for that MH step, \(a^\star\) is the target acceptance rate, and \(\eta\) is the adaptation step size (`adapt_step_size` in code). 

---

## Calling the sampler

The following is the standard usage pattern for calling SMC through the wrapper.

```python
import SMC as bpe
import IMsim as IMS

SIM_PARAMS = {
    "IMfolder": "IonMonger/",
    "param_option": "scan",     # "ImpZoo" for impedance
    "timeout": 40,      # Recommend 180s for impedance
}

SIM_INPUTS = {
    "scan_rate": scan_rates,    # Not necessary for impedance
    "input_types": input_type_Series,
    "input_iter_map": input_iter_Series,
    "input_const": input_const_Series,
}

simobj = IMS.SimIonMongerScan(SIM_PARAMS, SIM_INPUTS)   # SimIonMongerImp for impedance

# priors_array: shape (n_params, 2) = [[low, high], ...]
# priors_index: names aligned to priors_array rows
scaler_inputs = bpe.ScalingFuncs_input(priors_array, xindex=priors_index)
scaler_outputs = bpe.ScalingFuncs_output(y_target)

smc_params = {
    "jump_sigma": 1.0,
    "logl_denom": 0.00001,
    "N_particles": 500,
    "mh_steps": 4,
    "ess_thresh": 0.5,
    "cess_target_ratio": 0.99,
    "target_accept": 0.28,
    "adapt_step_size": 1.5,
    "proposal_scale": 0.7,
    "verbose": True,
    "n_workers": 15,
    "use_logit": True,
}

BPEobj = bpe.BPE_SMC(
    simobj=simobj,
    y_target=y_target,
    BPE_params=smc_params,
    priors=priors_array,
    scaler_inputs=scaler_inputs,
    scaler_outputs=scaler_outputs,
)

try:
    BPEobj.run()
finally:
    simobj.quit_matlab()

results = BPEobj.process_results()
```

---

## Multithreading and MATLAB engines (likelihood evaluation)

Likelihood evaluations across particles are parallelized using **Python multithreading** (`concurrent.futures.ThreadPoolExecutor`) when `log_likelihood(...)` is called on a **batch** of particles (`state.ndim == 2`). Particles are split into chunks and each chunk is evaluated in a worker thread.  

To keep simulator state isolated, each worker thread is mapped to a distinct simulator "engine" instance from `sim_scaled_list` (engine `k` evaluates chunk `k`). The effective number of threads is capped by the number of available simulator instances or a number of your choosing, e.g., 20.

**IonMonger simulators (`SimIonMongerScan`, `SimIonMongerImp`) start one MATLAB Engine session per simulator instance** (`matlab.start_matlab()` in `__init__`). This is why multi-engine mode matters: with `n_engines > 1`, you get multiple independent MATLAB sessions that can run concurrently. Each session also pins MATLAB's internal threading via `maxNumCompThreads(1)` to avoid CPU oversubscription when many engines run at once.  

Finally, each MATLAB call is launched with `background=True` and waited on with `future.result(timeout=...)` to enforce per-simulation timeouts; this async MATLAB call is separate from (and nested inside) the SMC multithreading. 

---

## Outputs

- `SMCTempering.run()` returns `(particles_scaled, weights, acceptance_rate, ess_final)`.   
- `BPE_SMC.process_results()` returns posterior samples and summary statistics in both scaled and unscaled spaces (e.g., `samples_unsc_DF`, `means_unsc`, `variances_unsc`, `weights`).
