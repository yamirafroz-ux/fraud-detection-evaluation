"""Exact OU transitions and exponential-kernel Hawkes thinning on constant-base intervals."""

import numpy as np


def ou_step(x, mean, kappa, sigma, dt, rng):
    if (
        not all(np.isfinite(v) for v in (kappa, sigma, dt))
        or not np.isfinite(x).all()
        or not np.isfinite(mean).all()
    ):
        raise ValueError("OU inputs must be finite")
    if kappa <= 0 or sigma < 0 or dt < 0:
        raise ValueError("Require kappa > 0, sigma >= 0, dt >= 0")
    a = np.exp(-kappa * dt)
    sd = sigma * np.sqrt(-np.expm1(-2 * kappa * dt) / (2 * kappa))
    return mean + (x - mean) * a + sd * rng.normal(size=np.shape(x))


def hawkes_interval(base, excitation, eta, beta, duration, rng, max_events=100000):
    """Return offsets and residual excitation; rejects safely instead of clipping intensity.

    lambda(t) = base + excitation*exp(-beta*t) + sum eta*beta*exp(-beta*(t-ti)).
    Base is constant in this interval. Carry residual excitation across boundaries.
    """
    if not all(np.isfinite(v) for v in (base, excitation, eta, beta, duration)):
        raise ValueError("Process parameters must be finite")
    if base < 0 or excitation < 0 or not 0 <= eta < 1 or beta <= 0 or duration < 0:
        raise ValueError("Invalid Hawkes parameters")
    t, z, times = 0.0, float(excitation), []
    while t < duration:
        upper = base + z
        if upper == 0:
            break
        if not np.isfinite(upper):
            raise RuntimeError("Intensity overflow; reduce process parameters")
        dt = rng.exponential(1 / upper)
        if t + dt <= t:
            raise RuntimeError("Event spacing below numerical resolution; reduce intensity")
        if t + dt >= duration:
            z *= np.exp(-beta * (duration - t))
            break
        z *= np.exp(-beta * dt)
        t += dt
        if rng.random() * upper < base + z:
            if len(times) >= max_events:
                raise RuntimeError("Event safety limit reached; lower rate or branching ratio")
            times.append(t)
            z += eta * beta
    return np.asarray(times), z


def gibbs_probabilities(energy, temperature):
    """Stable Boltzmann weights; temperature controls exploration versus preference."""
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1 or not energy.size:
        raise ValueError("Energy must be a nonempty vector")
    if temperature <= 0 or not np.isfinite(temperature) or not np.isfinite(energy).all():
        raise ValueError("Finite energy and positive finite temperature required")
    weights = np.exp(-(energy - energy.min()) / temperature)
    return weights / weights.sum()
