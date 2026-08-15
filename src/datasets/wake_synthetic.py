"""
wake_synthetic.py
=================

Offline stand-in for the April porous-disc wind-tunnel experiment
(/rds/general/project/immanuel/live/Seagate/april_experiment).

The point is *not* to be a good wake model. The point is to produce data with
the same shape, layout and pathologies as the real thing, while keeping a
ground truth you can assert against. Everything the real data hides -- the true
rank, the true sensor map, the true noise level -- is returned alongside the
data here, so a reconstruction method can be checked rather than eyeballed.

Construction
------------
The velocity field is built as an *exactly rank-r* expansion

    u(x, y, t) = u_mean(x, y) + sum_k a_k(t) psi_k(x, y)                    (1)

where the psi_k are physically-motivated wake structures (meandering,
breathing, shedding for each of the three discs) and the a_k are narrowband
oscillators. So the field looks like a disc-array wake, but its POD spectrum
has exactly r non-zero singular values -- which is a machine-precision
assertion, not a judgement call.

Sensors (six-component load cells under each disc) are generated from the same
a_k. In ``sensor_response="linear"`` mode

    s(t) = M_true @ a(t) + b + noise                                        (2)

so POD-LSE is *exactly* the right model and must recover the field to
round-off. In ``"quadratic"`` mode the thrust term is proportional to the
square of the disc-averaged incoming velocity, which is what a real load cell
measures. That puts an irreducible floor under any linear estimator -- the
concrete version of "forces and moments are nonlinear functionals of the flow
field" from Novoa's notes, and the reason the two-branch autoencoder exists.

Array layout follows the repo convention from ``datasets/snapshots.py``:

    U : (Nu, Nt, Nx, Ny)    NaN marks solid-body (disc) points
    S : (Nt, Ns)            sensor time series, Ns = n_discs * 6

Usage
-----
::

    from datasets.wake_synthetic import WakeConfig, generate

    case = generate(WakeConfig(n_t=2000, yaw_deg=(30.0, 15.0, 0.0)))
    U, S = case.U, case.S           # (2, 2000, Nx, Ny), (2000, 18)
    case.truth.rank                 # exact rank of U about its temporal mean

    # write an RDS-shaped run directory to disk and read it back
    from datasets.wake_synthetic import write_run, load_run
    write_run("data/synthetic/4p5d_10ms_yaw_30_15_0", case)
    case2 = load_run("data/synthetic/4p5d_10ms_yaw_30_15_0")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

__all__ = [
    "WakeConfig",
    "Truth",
    "SyntheticWake",
    "generate",
    "write_run",
    "load_run",
    "FORCE_COMPONENTS",
]

# order of the six load-cell channels under each disc
FORCE_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WakeConfig:
    """Everything that defines one synthetic run.

    Defaults follow the experiment on the slides: three porous discs spaced
    4.5 D apart, U = 10 m/s, Re_D = 33 000. With nu = 1.5e-5 m^2/s that fixes
    D ~ 0.05 m, which is what ``disc_diameter`` is set to.
    """

    # -- rig -------------------------------------------------------------------
    n_discs: int = 3
    spacing_D: float = 4.5  # streamwise gap between discs, in diameters
    disc_diameter: float = 0.0495  # m, from Re = U D / nu = 33000
    u_inf: float = 10.0  # m/s freestream
    yaw_deg: tuple = (0.0, 0.0, 0.0)  # per-disc yaw, matches folder names
    c_thrust: float = 0.75  # disc thrust coefficient

    # -- PIV grid --------------------------------------------------------------
    n_x: int = 192  # streamwise points
    n_y: int = 80  # cross-stream points
    x_lim_D: tuple = (-1.0, 16.0)  # streamwise extent, in diameters
    y_lim_D: tuple = (-3.0, 3.0)  # cross-stream extent, in diameters
    mask_discs: bool = True  # write NaN over the disc footprints

    # -- time ------------------------------------------------------------------
    n_t: int = 2000
    f_sample: float = 250.0  # Hz, the April experiment's PIV acquisition rate

    # -- unsteadiness ----------------------------------------------------------
    st_shedding: float = 0.17  # Strouhal number, f D / U
    st_meander: float = 0.015  # wake meandering, much slower
    st_breathing: float = 0.045
    mode_bandwidth: float = 0.06  # relative width of each oscillator
    convection_ratio: float = 0.7  # wake convection speed / u_inf

    # -- sensors ---------------------------------------------------------------
    sensor_response: str = "quadratic"  # "linear" | "quadratic"
    sensor_snr_db: float = 40.0  # additive white noise, per channel
    sensor_bias: float = 0.0  # constant offset, fraction of channel std
    sensor_drift: float = 0.0  # linear drift over the record, same units
    sensor_quantisation: int = 0  # ADC bits; 0 disables
    sensor_dropout: float = 0.0  # fraction of samples set to NaN

    # -- PIV noise -------------------------------------------------------------
    piv_snr_db: float = float("inf")  # per-vector noise; inf = clean
    piv_outlier_frac: float = 0.0  # fraction of vectors replaced by junk

    seed: int = 0
    dtype: str = "float32"

    def replace(self, **kwargs) -> "WakeConfig":
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)


@dataclass
class Truth:
    """Ground truth that the real experiment does not give you.

    Every field here is something a reconstruction method is *supposed* to
    recover, so tests can assert against it directly.
    """

    psi: np.ndarray  # (N_x, r) spatial modes used to build the field
    a: np.ndarray  # (r, N_t) their temporal coefficients
    rank: int  # exact rank of U about its temporal mean
    m_true: Optional[np.ndarray]  # (Ns, r) linear sensor map, if linear mode
    sensor_noise_std: np.ndarray  # (Ns,) std of the noise actually added
    mode_labels: list  # human-readable name per mode
    freqs: np.ndarray  # (r,) centre frequency of each mode, Hz


@dataclass
class SyntheticWake:
    U: np.ndarray  # (Nu, Nt, Nx, Ny) velocity fields, NaN at discs
    S: np.ndarray  # (Nt, Ns) sensor time series
    t: np.ndarray  # (Nt,) time, seconds
    x: np.ndarray  # (Nx,) streamwise coordinate, metres
    y: np.ndarray  # (Ny,) cross-stream coordinate, metres
    fluid_mask: np.ndarray  # (Nx, Ny) True where fluid
    channels: list  # Ns channel names, e.g. "disc0_Fx"
    cfg: WakeConfig
    truth: Truth

    # -- convenience ----------------------------------------------------------

    @property
    def n_t(self) -> int:
        return self.U.shape[1]

    def flat(self, subtract_mean: bool = True) -> np.ndarray:
        """The (N_fluid*Nu, N_t) data matrix Q that POD expects."""
        Nu, Nt = self.U.shape[0], self.U.shape[1]
        blocks = [
            self.U[i].reshape(Nt, -1)[:, self.fluid_mask.ravel()].T for i in range(Nu)
        ]
        Q = np.concatenate(blocks, axis=0)
        return Q - Q.mean(axis=1, keepdims=True) if subtract_mean else Q

    def to_grid(self, q: np.ndarray) -> np.ndarray:
        """Inverse of ``flat`` for one column: (N_fluid*Nu,) -> (Nu, Nx, Ny)."""
        Nu = self.U.shape[0]
        n_fluid = int(self.fluid_mask.sum())
        out = np.full((Nu, self.fluid_mask.size), np.nan, dtype=q.dtype)
        for i in range(Nu):
            out[i, self.fluid_mask.ravel()] = q[i * n_fluid : (i + 1) * n_fluid]
        return out.reshape((Nu,) + self.fluid_mask.shape)


# ── spatial building blocks ───────────────────────────────────────────────────


def _wake_deficit(X, Y, x0, y0, gamma, cfg: WakeConfig):
    """Bastankhah-style Gaussian deficit for one disc, normalised by u_inf.

    ``gamma`` is the yaw angle in radians; it both deflects the wake laterally
    and reduces the thrust, which is the leading-order effect seen in the
    yawed runs.
    """
    D = cfg.disc_diameter
    xi = np.maximum(X - x0, 0.0) / D  # downstream distance in diameters
    growth = 0.035  # wake expansion rate
    # sigma0 has to keep ct / (8 sigma^2) below 1, or the square root below
    # goes imaginary and the deficit saturates at 100% -- which shows up as a
    # centreline velocity of zero (and, once fluctuations are added, negative)
    # right behind the disc. sqrt(c_thrust / 8) is the floor; 0.33 clears it
    # for c_thrust up to ~0.87 and gives ~60% peak deficit at the disc.
    sigma = growth * xi + 0.33  # in diameters

    # lateral deflection of a yawed disc wake, small-angle far-wake form
    ct = cfg.c_thrust * np.cos(gamma) ** 2
    theta0 = 0.3 * gamma * (1.0 - np.sqrt(np.maximum(1.0 - ct, 0.0))) / np.cos(gamma)
    y_c = y0 + D * theta0 * xi

    amp = 1.0 - np.sqrt(np.maximum(1.0 - ct / (8.0 * sigma**2), 0.0))
    core = np.exp(-0.5 * ((Y - y_c) / (sigma * D)) ** 2)
    upstream = X >= x0 - 0.25 * D
    return amp * core * upstream, y_c, sigma * D


def _spatial_modes(X, Y, cfg: WakeConfig):
    """Build u_mean and the exactly-rank-r spatial mode set.

    Three structures per disc:
      meander   -- d(deficit)/dy, the response to lateral displacement
      breathe   -- d(deficit)/d(sigma), the response to width change
      shed x2   -- a convecting wave pair (sin/cos) in the shear layer

    The sin/cos pair is what lets a travelling structure live in a *real*
    two-dimensional subspace, which is exactly how a convecting wake shows up
    in POD -- as a pair of modes at equal energy in quadrature.
    """
    D = cfg.disc_diameter
    u_mean_y = np.zeros_like(X)
    deficits = []

    psis, labels, freqs = [], [], []
    f_shed = cfg.st_shedding * cfg.u_inf / D
    f_mean = cfg.st_meander * cfg.u_inf / D
    f_brea = cfg.st_breathing * cfg.u_inf / D
    lam = cfg.convection_ratio * cfg.u_inf / f_shed  # shedding wavelength

    yaws = list(cfg.yaw_deg) + [0.0] * cfg.n_discs
    for d in range(cfg.n_discs):
        x0 = d * cfg.spacing_D * D
        gamma = np.deg2rad(yaws[d])
        defc, y_c, sig = _wake_deficit(X, Y, x0, 0.0, gamma, cfg)
        deficits.append(defc)

        # meandering: lateral shift -> derivative of the deficit in y
        dy = (Y - y_c) / sig**2
        psis.append(np.stack([defc * dy, np.zeros_like(X)]))
        labels.append(f"disc{d}_meander")
        freqs.append(f_mean)

        # breathing: width change -> derivative wrt sigma
        psis.append(np.stack([defc * (((Y - y_c) / sig) ** 2 - 1.0), np.zeros_like(X)]))
        labels.append(f"disc{d}_breathe")
        freqs.append(f_brea)

        # shedding: convecting wave localised on the shear layers
        xi = np.maximum(X - x0, 0.0)
        env = np.exp(-0.5 * ((Y - y_c) / sig) ** 2) * np.exp(-xi / (6.0 * D))
        env = env * (X >= x0)
        phase = 2.0 * np.pi * xi / lam
        for name, wave in (("shed_c", np.cos(phase)), ("shed_s", np.sin(phase))):
            ux = env * wave * (Y - y_c) / sig
            uy = -env * wave
            psis.append(np.stack([ux, uy]))
            labels.append(f"disc{d}_{name}")
            freqs.append(f_shed)

    # Sum-of-squares deficit superposition (Katic 1986) rather than plain
    # addition. Adding deficits linearly double-counts where two wakes overlap:
    # a point 4.4 D downstream sits in disc 0's recovering wake *and* at the
    # face of disc 1, and the linear sum there exceeds 100% blockage, i.e. a
    # negative mean velocity. Only the mean flow is affected -- the mode
    # expansion, and so the exact rank, is untouched.
    u_mean_x = 1.0 - np.sqrt(np.sum(np.stack(deficits) ** 2, axis=0))
    u_mean = np.stack([u_mean_x * cfg.u_inf, u_mean_y * cfg.u_inf])
    Psi = np.stack(psis)  # (r, Nu, Nx, Ny)

    # Normalise each mode to unit peak. The shapes above are derivatives of the
    # deficit, so their raw magnitudes carry whatever the differentiation left
    # behind -- d/dy brings a 1/sigma^2 with sigma in metres, order 1e4 -- and
    # the amplitude knobs in _temporal_coeffs would mean nothing.
    #
    # Peak rather than RMS because these modes are spatially localised: a mode
    # confined to one wake has a domain RMS far below its local amplitude, so
    # normalising by RMS makes the *local* fluctuation many times larger than
    # the knob claims. Peak-normalised, "0.06 of u_inf" is the fluctuation you
    # actually get where the mode lives.
    peak = np.abs(Psi).max(axis=(1, 2, 3), keepdims=True)
    Psi = Psi / np.where(peak > 0, peak, 1.0)
    return u_mean, Psi, labels, np.asarray(freqs)


# ── temporal building blocks ──────────────────────────────────────────────────


def _narrowband(rng, n_t, dt, f0, bandwidth):
    """A unit-variance narrowband signal centred on f0.

    White noise shaped by a Gaussian window in the frequency domain. Genuinely
    stochastic -- so the reconstruction problem is not trivially periodic --
    but with a well-defined spectral peak, like real wake meandering.
    """
    w = rng.standard_normal(n_t)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n_t, dt)
    W *= np.exp(-0.5 * ((f - f0) / max(bandwidth * f0, f[1])) ** 2)
    s = np.fft.irfft(W, n=n_t)
    return s / (s.std() + 1e-300)


def _temporal_coeffs(cfg: WakeConfig, freqs, labels, rng):
    """Coefficients a_k(t), with the physical couplings a wind farm row has.

    Two couplings are put in deliberately, because they are what makes the
    sparse-sensor problem interesting rather than trivially separable:

      * a shed_c/shed_s pair shares one signal in quadrature, so the pair
        behaves as one convecting structure rather than two free modes;
      * a downstream disc inherits a lagged copy of the disc in front of it,
        which is the actual mechanism the experiment is set up to study.
    """
    n_t, dt = cfg.n_t, 1.0 / cfg.f_sample
    D = cfg.disc_diameter
    a = np.zeros((len(freqs), n_t))

    lag = int(round(cfg.spacing_D * D / (cfg.convection_ratio * cfg.u_inf) / dt))
    by_name = {lab: i for i, lab in enumerate(labels)}

    for k, (lab, f0) in enumerate(zip(labels, freqs)):
        if lab.endswith("shed_s"):
            continue  # filled in below, as the quadrature partner of shed_c
        base = _narrowband(rng, n_t, dt, f0, cfg.mode_bandwidth)

        disc = int(lab[4])
        if disc > 0:  # inherit from the disc in front
            up = lab.replace(f"disc{disc}", f"disc{disc - 1}")
            if up in by_name:
                shifted = np.roll(a[by_name[up]], lag * disc)
                shifted[: lag * disc] = 0.0
                base = 0.65 * base + 0.55 * shifted
                base /= base.std() + 1e-300
        a[k] = base

        if lab.endswith("shed_c"):
            # Hilbert transform -> exact 90-degree partner, no scipy needed
            A = np.fft.rfft(base)
            A[1:-1] *= -1j
            part = np.fft.irfft(A, n=n_t)
            a[by_name[lab.replace("shed_c", "shed_s")]] = part / (part.std() + 1e-300)

    # amplitudes as a fraction of u_inf, at each mode's own peak. Meandering
    # carries most of the unsteady energy in a disc wake, shedding the least.
    scale = np.array(
        [
            0.060 if "meander" in l else 0.030 if "breathe" in l else 0.020
            for l in labels
        ]
    )
    # discs further back sit in a more turbulent inflow
    scale = scale * np.array([1.0 + 0.35 * int(l[4]) for l in labels])
    return a * scale[:, None] * cfg.u_inf


# ── sensors ───────────────────────────────────────────────────────────────────


def _load_cell_weights(X, Y, cfg: WakeConfig):
    """Spatial weight and component pair for each of the six channels per disc.

    A six-axis load cell does not measure one number six times -- it resolves
    the first few *moments* of the load distribution across the disc face. So
    the six channels are modelled as the uniform, linear and quadratic
    weightings in the radial coordinate eta = y / (D/2), applied to the
    streamwise (u*u) and cross (u*v) load contributions:

        Fx <- w0 u u     Fy <- w0 u v
        Fz <- w1 u u     Mx <- w1 u v
        My <- w2 u u     Mz <- w2 u v

    That matters for more than realism. If the six channels were scaled copies
    of one another -- as they are if every channel just averages u over the
    disc -- the load cells would span a 2-dimensional subspace no matter how
    many of them you bolt on, and no estimator, linear or otherwise, could
    recover a higher-rank field. Independent weightings are what make the
    sensor array actually observe the state.

    Returns (W, pairs) with W (n_discs*6, Nx*Ny) and pairs the (i, j) velocity
    component indices each channel multiplies together.
    """
    D = cfg.disc_diameter
    eta = Y / (0.5 * D)
    profiles = (np.ones_like(eta), eta, 1.5 * eta**2 - 0.5)  # 0th, 1st, 2nd moment
    pairs_per_channel = ((0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1))
    profile_index = (0, 0, 1, 1, 2, 2)

    W, pairs = [], []
    for d in range(cfg.n_discs):
        x0 = d * cfg.spacing_D * D
        # A load cell does not sample a razor-thin slab at the disc plane: the
        # load responds to the induction region ahead of the disc and the near
        # wake behind it, order 1 D either way. Modelling it as a thin slab is
        # not merely less realistic, it is degenerate -- a convecting structure
        # has phase 2 pi xi / lambda, so at xi = 0 the sine component is
        # identically zero and one of every shedding pair becomes invisible to
        # every sensor. A kernel that spans a decent fraction of a wavelength
        # sees both components.
        foot = np.exp(-0.5 * ((X - x0) / (0.5 * D)) ** 2) * (np.abs(Y) < 0.6 * D)
        area = foot.sum()
        if area <= 0:
            raise ValueError(
                f"disc {d} footprint is empty; the grid is too coarse for "
                f"n_x={cfg.n_x}, n_y={cfg.n_y}"
            )
        for c in range(6):
            W.append((foot * profiles[profile_index[c]] / area).ravel())
            pairs.append(pairs_per_channel[c])
    return np.stack(W), pairs


def _make_sensors(u_mean, Psi, a, X, Y, cfg: WakeConfig, rng):
    """Six-component load-cell signals for each disc.

    Every channel is a weighted integral of a product of two velocity
    components -- a load is a dynamic pressure times an area, so it is
    quadratic in velocity by construction. Substituting the modal expansion
    u = u_mean + sum_k a_k psi_k gives, exactly,

        s_c(t) = c0_c + sum_k c1_ck a_k(t) + sum_kl c2_ckl a_k(t) a_l(t)

    which is expanded analytically rather than by squaring fields snapshot by
    snapshot: it is both cheaper and exact. ``sensor_response="linear"`` keeps
    the first two terms, which is the linearisation about the mean flow;
    ``"quadratic"`` keeps all three, which is what a real load cell gives you.

    Returns (S, M_true, noise_std, channels). ``M_true`` is not None only in
    linear mode, where the map really is linear and a least-squares estimator
    can be asked to reproduce it exactly.
    """
    if cfg.sensor_response not in ("linear", "quadratic"):
        raise ValueError(
            f"sensor_response must be 'linear' or 'quadratic', "
            f"got {cfg.sensor_response!r}"
        )
    r = Psi.shape[0]
    W, pairs = _load_cell_weights(X, Y, cfg)  # (Ns, Nx*Ny)
    n_ch = W.shape[0]

    um = u_mean.reshape(u_mean.shape[0], -1)  # (Nu, Nx*Ny)
    pm = Psi.reshape(r, Psi.shape[1], -1)  # (r, Nu, Nx*Ny)

    rho, D = 1.225, cfg.disc_diameter
    gain = 0.5 * rho * (0.25 * np.pi * D**2) * cfg.c_thrust

    c0 = np.empty(n_ch)
    c1 = np.empty((n_ch, r))
    c2 = np.empty((n_ch, r, r))
    for c, (i, j) in enumerate(pairs):
        w = W[c]
        c0[c] = w @ (um[i] * um[j])
        c1[c] = pm[:, i] @ (w * um[j]) + pm[:, j] @ (w * um[i])
        c2[c] = np.einsum("kn,ln,n->kl", pm[:, i], pm[:, j], w)
        c2[c] = 0.5 * (c2[c] + c2[c].T)  # a_k a_l is symmetric; so must c2 be

    c0, c1, c2 = gain * c0, gain * c1, gain * c2

    S = (c0[:, None] + c1 @ a).T  # (n_t, Ns), the linear model
    if cfg.sensor_response == "quadratic":
        S = S + np.einsum("ckl,kt,lt->tc", c2, a, a)
    M_true = c1  # (Ns, r)
    channels = [
        f"disc{d}_{c}" for d in range(cfg.n_discs) for c in FORCE_COMPONENTS
    ]

    # -- contamination --------------------------------------------------------
    sig = S.std(axis=0)
    sig = np.where(sig > 0, sig, 1.0)
    noise_std = sig * 10.0 ** (-cfg.sensor_snr_db / 20.0)
    S = S + rng.standard_normal(S.shape) * noise_std

    if cfg.sensor_bias:
        S = S + cfg.sensor_bias * sig * rng.standard_normal(S.shape[1])
    if cfg.sensor_drift:
        ramp = np.linspace(0.0, 1.0, S.shape[0])[:, None]
        S = S + cfg.sensor_drift * sig * ramp * rng.standard_normal(S.shape[1])
    if cfg.sensor_quantisation:
        # symmetric mid-tread quantiser at +-4 sigma
        full = 4.0 * sig
        step = 2.0 * full / (2**cfg.sensor_quantisation - 1)
        S = np.round(S / step) * step
    if cfg.sensor_dropout:
        drop = rng.random(S.shape) < cfg.sensor_dropout
        S = np.where(drop, np.nan, S)

    return S, (M_true if cfg.sensor_response == "linear" else None), noise_std, channels


# ── the generator ─────────────────────────────────────────────────────────────


def generate(cfg: WakeConfig = WakeConfig()) -> SyntheticWake:
    """Build one synthetic run. Deterministic in ``cfg.seed``."""
    if len(cfg.yaw_deg) < cfg.n_discs:
        raise ValueError(
            f"yaw_deg has {len(cfg.yaw_deg)} entries but n_discs={cfg.n_discs}"
        )
    rng = np.random.default_rng(cfg.seed)
    D = cfg.disc_diameter

    x = np.linspace(cfg.x_lim_D[0], cfg.x_lim_D[1], cfg.n_x) * D
    y = np.linspace(cfg.y_lim_D[0], cfg.y_lim_D[1], cfg.n_y) * D
    X, Y = np.meshgrid(x, y, indexing="ij")  # (Nx, Ny)

    u_mean, Psi, labels, freqs = _spatial_modes(X, Y, cfg)
    a = _temporal_coeffs(cfg, freqs, labels, rng)

    # equation (1): mean + rank-r fluctuation
    r = Psi.shape[0]
    U = u_mean[:, None] + np.einsum("kt,kuxy->utxy", a, Psi)

    S, M_true, noise_std, channels = _make_sensors(u_mean, Psi, a, X, Y, cfg, rng)

    # -- PIV contamination ----------------------------------------------------
    if np.isfinite(cfg.piv_snr_db):
        amp = np.nanstd(U) * 10.0 ** (-cfg.piv_snr_db / 20.0)
        U = U + rng.standard_normal(U.shape) * amp
    if cfg.piv_outlier_frac:
        bad = rng.random(U.shape) < cfg.piv_outlier_frac
        U = np.where(bad, U + rng.standard_normal(U.shape) * 3.0 * cfg.u_inf, U)

    # -- disc footprints ------------------------------------------------------
    fluid = np.ones(X.shape, dtype=bool)
    if cfg.mask_discs:
        for d in range(cfg.n_discs):
            x0 = d * cfg.spacing_D * D
            fluid &= ~((np.abs(X - x0) < 0.06 * D) & (np.abs(Y) < 0.5 * D))
        U = np.where(fluid[None, None], U, np.nan)

    dt = 1.0 / cfg.f_sample
    dtype = np.dtype(cfg.dtype)
    return SyntheticWake(
        U=U.astype(dtype),
        S=S.astype(dtype),
        t=(np.arange(cfg.n_t) * dt).astype(np.float64),
        x=x,
        y=y,
        fluid_mask=fluid,
        channels=channels,
        cfg=cfg,
        truth=Truth(
            psi=Psi.reshape(r, -1)[:, np.tile(fluid.ravel(), Psi.shape[1])].T,
            a=a,
            rank=r,
            m_true=M_true,
            sensor_noise_std=noise_std,
            mode_labels=labels,
            freqs=freqs,
        ),
    )


# ── RDS-shaped IO ─────────────────────────────────────────────────────────────
#
# The real runs store one .npz per snapshot under piv_snapshots_highres/, and
# the load cells under synced_forces/. The *key names inside* those .npz files
# are a guess -- see docs/sparse_sensors/README.md, "Reading the real data",
# for the three lines that tell you the actual ones. Everything downstream of
# `load_run` only depends on the returned arrays, so adapting is a one-function
# change if the guess is wrong.

_PIV_KEYS = ("u", "v", "x", "y")


def write_run(path: str, case: SyntheticWake, keys=_PIV_KEYS) -> str:
    """Write ``case`` as a run directory shaped like the RDS ones."""
    piv_dir = os.path.join(path, "piv_snapshots_highres")
    frc_dir = os.path.join(path, "synced_forces")
    os.makedirs(piv_dir, exist_ok=True)
    os.makedirs(frc_dir, exist_ok=True)

    ku, kv, kx, ky = keys
    for i in range(case.n_t):
        np.savez_compressed(
            os.path.join(piv_dir, f"snapshot_{i:06d}.npz"),
            **{ku: case.U[0, i], kv: case.U[1, i], kx: case.x, ky: case.y},
        )
    np.savez_compressed(
        os.path.join(frc_dir, "forces.npz"),
        t=case.t,
        F=case.S,
        channels=np.array(case.channels),
    )
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(asdict(case.cfg), f, indent=2)
    with open(os.path.join(path, "readme.md"), "w") as f:
        f.write(
            f"# synthetic run\n\n"
            f"Generated by `datasets.wake_synthetic`. NOT experimental data.\n\n"
            f"- {case.n_t} snapshots, {case.U.shape[2]}x{case.U.shape[3]} grid\n"
            f"- {len(case.channels)} force channels at {case.cfg.f_sample} Hz\n"
            f"- exact rank of the fluctuation field: {case.truth.rank}\n"
            f"- sensor response: {case.cfg.sensor_response}\n"
        )
    return path


def load_run(path: str, keys=_PIV_KEYS, max_snapshots: Optional[int] = None):
    """Read a run directory back. Returns (U, S, t, x, y, channels).

    Deliberately returns plain arrays rather than a ``SyntheticWake``: this is
    the function you repoint at the real RDS data, which has no ground truth
    to put in a ``Truth``.
    """
    ku, kv, kx, ky = keys
    piv_dir = os.path.join(path, "piv_snapshots_highres")
    files = sorted(f for f in os.listdir(piv_dir) if f.endswith(".npz"))
    if max_snapshots:
        files = files[:max_snapshots]
    if not files:
        raise FileNotFoundError(f"no .npz snapshots in {piv_dir}")

    with np.load(os.path.join(piv_dir, files[0])) as z:
        x, y = z[kx], z[ky]

    us, vs = [], []
    for fn in files:
        with np.load(os.path.join(piv_dir, fn)) as z:
            us.append(z[ku])
            vs.append(z[kv])
    U = np.stack([np.stack(us), np.stack(vs)])  # (Nu, Nt, Nx, Ny)

    with np.load(os.path.join(path, "synced_forces", "forces.npz")) as z:
        t, S, channels = z["t"], z["F"], [str(c) for c in z["channels"]]

    n = U.shape[1]
    return U, S[:n], t[:n], x, y, channels
