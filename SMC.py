import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------------------------------------------
# Scaling helpers
# -----------------------------------------------------------------------------
class ScalingFuncs_input:
    """Scale/unscale model inputs using per-parameter [min, max] bounds."""

    def __init__(self, scaling_array, xindex):
        self.scaling_array = np.asarray(scaling_array, dtype=float)
        self.xindex = xindex

    def scale(self, x):
        x = np.asarray(x, dtype=float)
        lo = self.scaling_array[:, 0]
        hi = self.scaling_array[:, 1]
        return (x - lo) / (hi - lo)

    def unscale(self, x_scaled):
        x_scaled = np.asarray(x_scaled, dtype=float)
        lo = self.scaling_array[:, 0]
        hi = self.scaling_array[:, 1]
        x = lo + x_scaled * (hi - lo)
        return pd.Series(x, index=self.xindex)

    def unscale_DF(self, x_scaled):
        x_scaled_df = pd.DataFrame(x_scaled, columns=self.xindex)
        lo = pd.Series(self.scaling_array[:, 0], index=self.xindex)
        hi = pd.Series(self.scaling_array[:, 1], index=self.xindex)
        return lo + x_scaled_df * (hi - lo)


class ScalingFuncs_output:
    """Scale/unscale model outputs using a per-output scale factor."""

    def __init__(self, scaling_array):
        self.scaling_array = np.asarray(scaling_array, dtype=float)

    def scale(self, y):
        return np.asarray(y, dtype=float) / self.scaling_array

    def unscale(self, y_scaled):
        return np.asarray(y_scaled, dtype=float) * self.scaling_array


class Sim_scaled:
    """Adapter: runs the underlying simulator in unscaled space, returns scaled outputs."""

    def __init__(self, sim, scaler_inputs: ScalingFuncs_input, scaler_outputs: ScalingFuncs_output):
        self.sim = sim
        self.scaler_inputs = scaler_inputs
        self.scaler_outputs = scaler_outputs

    def run(self, x_scaled_input, return_detailed_results=False):
        x_input = self.scaler_inputs.unscale(x_scaled_input)

        if return_detailed_results:
            y, ydetail = self.sim.run(x_input, return_detailed_results)
            return self.scaler_outputs.scale(y), ydetail

        y = self.sim.run(x_input, return_detailed_results)
        return self.scaler_outputs.scale(y)


# -----------------------------------------------------------------------------
# SMC Tempering
# -----------------------------------------------------------------------------
class SMCTempering:
    """SMC tempering with MH rejuvenation and optional logit reparam."""

    def __init__(self, sim_scaled, prior_ranges, y, smc_params, verbose=True):
        # Support either a single Sim_scaled or a list (one per engine/thread).
        if isinstance(sim_scaled, (list, tuple)):
            if not sim_scaled:
                raise ValueError("sim_scaled list cannot be empty.")
            self.sim_scaled_list = list(sim_scaled)
        else:
            self.sim_scaled_list = [sim_scaled]

        self.sim_scaled = self.sim_scaled_list[0]
        self.prior_ranges = np.asarray(prior_ranges, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.dim = self.prior_ranges.shape[0]

        # Core params
        self.N = int(smc_params["N_particles"])
        self.mh_steps = int(smc_params["mh_steps"])
        self.logl_denom = float(smc_params["logl_denom"])

        self.ess_thresh = float(smc_params["ess_thresh"])
        self.cess_target_ratio = float(smc_params["cess_target_ratio"])

        self.jump_sigma = float(smc_params["jump_sigma"])
        self.jump_dist_sigmas = self.jump_sigma * np.ones(self.dim, dtype=float)

        self.target_accept = float(smc_params["target_accept"])
        self.adapt_accept = bool(smc_params.get("adapt_accept", True))
        self.adapt_step_size = float(smc_params["adapt_step_size"])
        self._proposal_scale = float(smc_params["proposal_scale"])

        self.verbose = bool(verbose)

        # Engines / workers for likelihood evaluation
        requested = smc_params.get("n_engines", smc_params.get("n_workers", 1))
        self.n_workers = max(1, int(requested))
        self.n_workers = min(self.n_workers, 20, len(self.sim_scaled_list))

        # Optional logit reparam
        self.use_logit = bool(smc_params.get("use_logit", False))
        self._prior_low = self.prior_ranges[:, 0].astype(float)
        self._prior_high = self.prior_ranges[:, 1].astype(float)
        self._prior_range = self._prior_high - self._prior_low

        # Proposal state
        self._base_proposal_cov = np.diag(self.jump_dist_sigmas**2)
        self._proposal_cov = None
        self._proposal_chol = None
        self._set_proposal_from_cov(self._base_proposal_cov)

    # ----------------------------- utilities ---------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _ess(weights):
        weights = np.asarray(weights, dtype=float)
        return 1.0 / np.sum(weights**2)

    # ------------------------- logit reparameterisation -----------------------
    @staticmethod
    def _inv_logit(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _logit(u, eps=1e-12):
        u = np.asarray(u, dtype=float)
        u = np.clip(u, eps, 1.0 - eps)
        return np.log(u) - np.log(1.0 - u)

    def _to_internal(self, x_scaled):
        if not self.use_logit:
            return np.asarray(x_scaled, dtype=float)
        u = (np.asarray(x_scaled, dtype=float) - self._prior_low) / self._prior_range
        return self._logit(u)

    def _from_internal(self, state):
        if not self.use_logit:
            return np.asarray(state, dtype=float)
        u = self._inv_logit(np.asarray(state, dtype=float))
        return self._prior_low + self._prior_range * u

    # ---------------------- proposal covariance handling ----------------------
    def _set_proposal_from_cov(self, cov):
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        d = self.dim
        base = (2.38**2) / d
        scale = base * (self._proposal_scale**2)

        cov_scaled = scale * cov + 1e-10 * np.eye(d)
        try:
            chol = np.linalg.cholesky(cov_scaled)
        except np.linalg.LinAlgError:
            cov_scaled = cov_scaled + 1e-6 * np.eye(d)
            chol = np.linalg.cholesky(cov_scaled)

        self._proposal_cov = cov_scaled
        self._proposal_chol = chol

    def _adapt_proposal_scale(self, accept_rate_step):
        if not self.adapt_accept or not np.isfinite(accept_rate_step):
            return

        log_s = np.log(self._proposal_scale)
        log_s += self.adapt_step_size * (accept_rate_step - self.target_accept)
        log_s = np.clip(log_s, np.log(1e-3), np.log(1e3))
        new_scale = float(np.exp(log_s))

        factor = new_scale / self._proposal_scale
        self._proposal_cov *= factor**2
        self._proposal_chol *= factor
        self._proposal_scale = new_scale

        self._log(
            f"Adapted proposal scale to {self._proposal_scale:.4f} "
            f"(step accept {accept_rate_step:.3f}, target {self.target_accept:.3f})"
        )

    def _update_adaptive_proposal(self, particles, weights=None):
        X = np.atleast_2d(np.asarray(particles, dtype=float))
        n = X.shape[0]
        if n <= 1:
            self._set_proposal_from_cov(self._base_proposal_cov)
            return

        if weights is None:
            cov = np.cov(X, rowvar=False)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            w = w / np.sum(w)
            mean = np.average(X, axis=0, weights=w)
            diff = X - mean
            cov = diff.T @ (diff * w[:, None])

        self._set_proposal_from_cov(cov)
        self._log("Updated adaptive proposal covariance.")

    # ------------------------------- prior -----------------------------------
    def initial_sample(self):
        x = np.random.uniform(self._prior_low, self._prior_high, size=(self.N, self.dim))
        return self._to_internal(x)

    def log_prior(self, state):
        state = np.atleast_2d(np.asarray(state, dtype=float))
        x = self._from_internal(state)

        in_bounds = np.all((x >= self._prior_low) & (x <= self._prior_high), axis=1)
        if not self.use_logit:
            return np.where(in_bounds, 0.0, -np.inf)

        u = (x - self._prior_low) / self._prior_range
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        lp = np.sum(np.log(u) + np.log(1.0 - u), axis=1)
        lp[~in_bounds] = -np.inf
        return lp

    # ----------------------------- likelihood --------------------------------
    def _single_log_likelihood_core(self, x_scaled, sim_scaled):
        try:
            outputs = sim_scaled.run(x_scaled)
            mse = np.mean((outputs - self.y) ** 2)
            return -0.5 * mse / self.logl_denom
        except Exception:
            return -np.inf

    def _split_indices(self, n_particles, n_workers):
        n_workers = min(n_workers, n_particles)
        base = n_particles // n_workers
        idx_slices = []
        start = 0
        for i in range(n_workers):
            end = start + base
            if i == n_workers - 1:
                end = n_particles
            idx_slices.append(np.arange(start, end))
            start = end
        return idx_slices

    def _loglikelihood_chunk(self, engine_idx, states, indices):
        sim = self.sim_scaled_list[engine_idx]
        out = np.empty(len(indices), dtype=float)
        for j, idx in enumerate(indices):
            x_scaled = self._from_internal(states[idx])
            out[j] = self._single_log_likelihood_core(x_scaled, sim)
        return indices, out

    def log_likelihood(self, state):
        state = np.asarray(state, dtype=float)

        if state.ndim == 1:
            x_scaled = self._from_internal(state)
            return self._single_log_likelihood_core(x_scaled, self.sim_scaled)

        n_particles = state.shape[0]
        if self.n_workers <= 1 or n_particles < 2:
            return np.array(
                [self._single_log_likelihood_core(self._from_internal(s), self.sim_scaled) for s in state]
            )

        n_workers = min(self.n_workers, len(self.sim_scaled_list), n_particles)
        index_slices = self._split_indices(n_particles, n_workers)
        results = np.empty(n_particles, dtype=float)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(self._loglikelihood_chunk, engine_idx, state, idx_chunk)
                for engine_idx, idx_chunk in enumerate(index_slices)
                if idx_chunk.size
            ]
            for fut in futures:
                idx_chunk, vals = fut.result()
                results[idx_chunk] = vals

        return results

    # ------------------------------ proposals --------------------------------
    def proposal_distribution(self, current_state):
        X = np.asarray(current_state, dtype=float)
        was_1d = X.ndim == 1
        X = np.atleast_2d(X)
        n, d = X.shape

        noise = np.random.normal(size=(n, d))
        steps = noise @ self._proposal_chol.T
        Xp = X + steps

        return Xp[0] if was_1d else Xp

    # ------------------------------ resampling --------------------------------
    @staticmethod
    def resample(weights, method="systematic", rng=None):
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        N = w.size
        rng = np.random if rng is None else rng

        if method == "stratified":
            positions = (rng.rand(N) + np.arange(N)) / N
        elif method == "systematic":
            u0 = rng.rand() / N
            positions = u0 + np.arange(N) / N
        else:
            raise ValueError("method must be 'stratified' or 'systematic'")

        cdf = np.cumsum(w)
        cdf[-1] = 1.0
        return np.searchsorted(cdf, positions, side="left")

    # --------------------------- tempering schedule ---------------------------
    def _compute_delta_lambda_root(self, log_liks, weights, lambda_curr):
        ll = np.asarray(log_liks, dtype=float)
        w = np.asarray(weights, dtype=float)

        mask = np.isfinite(ll) & np.isfinite(w)
        if not np.any(mask):
            return 1.0 - lambda_curr

        ll = ll[mask]
        w = w[mask]
        w = w / np.sum(w)

        N_eff = w.size
        remaining = 1.0 - lambda_curr
        if remaining <= 0.0:
            return 0.0

        target_cess = self.cess_target_ratio * N_eff

        def cess(delta):
            a = delta * ll
            a_max = np.max(a)
            v = np.exp(a - a_max)
            num = np.dot(w, v)
            denom = np.dot(w, v * v)
            return 0.0 if denom <= 0.0 else N_eff * (num * num) / denom

        if cess(remaining) >= target_cess:
            return remaining

        lo, hi = 0.0, remaining
        tol = 1e-3
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            c = cess(mid)
            if abs(c - target_cess) <= tol * N_eff:
                return mid
            if c > target_cess:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------- MH kernel --------------------------------
    def _mh_kernel(self, particles, log_liks, lambda_next, n_steps):
        curr_dist = np.asarray(particles, dtype=float)
        logLiks = np.asarray(log_liks, dtype=float)

        acc_total = 0
        prop_total = 0

        for k in range(n_steps):
            self._log(f"MH step {k + 1}/{n_steps}")

            prop_dist = self.proposal_distribution(curr_dist)

            logPrior_curr = self.log_prior(curr_dist)
            logPrior_prop = self.log_prior(prop_dist)

            logLiks_prop = self.log_likelihood(prop_dist)

            post_curr = lambda_next * logLiks + logPrior_curr
            post_prop = lambda_next * logLiks_prop + logPrior_prop

            dpost = post_prop - post_curr
            valid = np.isfinite(dpost)

            log_alpha = np.minimum(0.0, dpost)
            u = np.random.uniform(size=curr_dist.shape[0])

            accept = (np.log(u) < log_alpha) & valid

            n_acc = int(np.sum(accept))
            acc_total += n_acc
            prop_total += curr_dist.shape[0]

            curr_dist[accept] = prop_dist[accept]
            logLiks[accept] = logLiks_prop[accept]

            step_acc = n_acc / curr_dist.shape[0]
            self._log(f"MH accepted {n_acc}/{curr_dist.shape[0]} ({step_acc:.3f})")
            self._adapt_proposal_scale(step_acc)

        return curr_dist, logLiks, acc_total, prop_total
    # --------------------------------- run -----------------------------------
    def run_smc(self):
        self._log("Starting SMC")

        particles = self.initial_sample()
        weights = np.ones(self.N) / self.N
        logweights = np.zeros(self.N, dtype=float)
        lambda_curr = 0.0

        acc_count = 0
        total_prop = 0

        log_liks = self.log_likelihood(particles)
        self._log("Initial log-likelihoods: " + ", ".join(f"{x:.7f}" for x in log_liks[:5]) + " ...")

        while lambda_curr < 1.0:
            delta = self._compute_delta_lambda_root(log_liks, weights, lambda_curr)
            if delta <= 0.0:
                self._log("Delta=0: resampling and rejuvenating at current lambda.")
                self._update_adaptive_proposal(particles, weights)

                idx = self.resample(weights, method="systematic")
                particles = particles[idx]
                log_liks = log_liks[idx]

                weights[:] = 1.0 / self.N
                logweights[:] = 0.0

                particles, log_liks, a, p = self._mh_kernel(particles, log_liks, lambda_curr, self.mh_steps)
                acc_count += a
                total_prop += p
                continue

            lambda_next = lambda_curr + delta
            self._log(f"Delta: {delta:.7f}, lambda_next: {lambda_next:.7f}")

            incr = delta * log_liks
            incr = np.where(np.isfinite(incr), incr, -np.inf)
            logweights = logweights + incr

            m = np.max(logweights[np.isfinite(logweights)])
            w_unnorm = np.exp(logweights - m)
            w_unnorm[~np.isfinite(w_unnorm)] = 0.0
            weights = w_unnorm / np.sum(w_unnorm)

            ess = self._ess(weights)
            self._log(f"ESS: {ess:.7f}")

            if ess < self.ess_thresh * self.N:
                self._log("Resampling (ESS below threshold).")
                self._update_adaptive_proposal(particles, weights)

                idx = self.resample(weights, method="systematic")
                particles = particles[idx]
                log_liks = log_liks[idx]

                weights[:] = 1.0 / self.N
                logweights[:] = 0.0

                particles, log_liks, a, p = self._mh_kernel(particles, log_liks, lambda_next, self.mh_steps)
                acc_count += a
                total_prop += p

            lambda_curr = lambda_next

        acc_rate = acc_count / total_prop if total_prop else np.nan
        ess_final = self._ess(weights)

        self._log("Lambda reached 1.0, stopping SMC.")
        self._log(f"Final acceptance rate: {acc_rate:.7f}")
        self._log(f"Final ESS: {ess_final:.7f}")

        particles_scaled = self._from_internal(particles)
        return particles_scaled, weights, acc_rate, ess_final

    def run(self):
        self._log("Initialising SMC")
        np.random.seed()
        return self.run_smc()


# -----------------------------------------------------------------------------
# BPE wrapper
# -----------------------------------------------------------------------------
class BPE_SMC:
    """Thin wrapper that builds SMCTempering and post-processes results."""

    def __init__(self, simobj, y_target, BPE_params, priors, scaler_inputs, scaler_outputs, ref_data=None):
        self.simobj = simobj
        self.y_target = np.asarray(y_target, dtype=float)
        self.BPE_params = dict(BPE_params)
        self.priors = np.asarray(priors, dtype=float)
        self.scaler_inputs = scaler_inputs
        self.scaler_outputs = scaler_outputs
        self.ref_data = ref_data

        # Prior ranges in scaled space
        self.prior_ranges_scaled = np.column_stack(
            (
                self.scaler_inputs.scale(self.priors[:, 0]),
                self.scaler_inputs.scale(self.priors[:, 1]),
            )
        )

        # Target in scaled output space
        self.y_scaled = self.scaler_outputs.scale(self.y_target)

        # Build simulator pool (optional)
        self.n_engines = int(self.BPE_params.get("n_engines", 1))
        simobjs_for_pool = [self.simobj]
        self._extra_simobjs = []

        if self.n_engines > 1:
            sim_cls = type(self.simobj)
            runparams = getattr(self.simobj, "runparams", None)
            runinputs = getattr(self.simobj, "runinputs", None)

            if runparams is not None and runinputs is not None:
                for _ in range(self.n_engines - 1):
                    sim_i = sim_cls(runparams, runinputs)
                    simobjs_for_pool.append(sim_i)
                    self._extra_simobjs.append(sim_i)
            else:
                print("WARNING: simobj has no runparams/runinputs; falling back to single-engine mode.")
                self.n_engines = 1

        sim_scaled_list = [Sim_scaled(s, self.scaler_inputs, self.scaler_outputs) for s in simobjs_for_pool]
        sim_scaled_arg = sim_scaled_list if len(sim_scaled_list) > 1 else sim_scaled_list[0]

        self.BPEobject = SMCTempering(
            sim_scaled_arg,
            self.prior_ranges_scaled,
            self.y_scaled,
            self.BPE_params,
            verbose=self.BPE_params.get("verbose", True),
        )

        self.particles = None
        self.weights = None
        self.acceptance_rate = None
        self.ess_final = None

    def run(self):
        try:
            self.particles, self.weights, self.acceptance_rate, self.ess_final = self.BPEobject.run()
        finally:
            for sim in self._extra_simobjs:
                quit_m = getattr(sim, "quit_matlab", None)
                if quit_m is not None:
                    try:
                        quit_m()
                    except Exception as e:
                        print(f"Error quitting extra MATLAB engine: {e}")
            self._extra_simobjs = []

    def process_results(self):
        print("Processing SMC results")

        particles_unsc_df = self.scaler_inputs.unscale_DF(self.particles)
        samples_scaled = self.particles
        weights = np.asarray(self.weights, dtype=float)

        means_unsc = self.scaler_inputs.unscale(np.mean(samples_scaled, axis=0))
        variances_unsc = np.var(particles_unsc_df, axis=0)

        cov_unsc_df = particles_unsc_df.cov()
        corr_unsc_df = particles_unsc_df.corr()

        print("Results processing completed")
        return {
            "means_unsc": means_unsc,
            "variances_unsc": variances_unsc,
            "samples_unsc_DF": particles_unsc_df,
            "acceptance_rate": self.acceptance_rate,
            "samples_scaled": samples_scaled,
            "weights": weights,
            "cov_unsc": cov_unsc_df,
            "corr_unsc": corr_unsc_df,
        }
