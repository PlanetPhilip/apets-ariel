"""Hardware-side port of hunger_games' real_na_cpg controller (controllers/real_na_cpg.py: the NA-CPG as published,
Tong et al. 2022), numpy only.

The training-side controllers (real_na_cpg_grid_w1_*_hw, with mirror pairing, locked phases, sign flips and the
hardware clamp of A, b and ha) are all just a parameterisation of the same CPG. The exporter
(hunger_games/src/export_real_archive.py) expands them into the plain per-joint vectors (phase, w, amplitudes, ha, b),
already clamped and permuted into this runtime's actuator order, so this class only needs the oscillator equations.
`weights` = those five groups concatenated; `initial_xy` is the seeded starting state of each oscillator, baked in
so no RNG has to match."""
import numpy as np

from .abstract import Controller
from ..mujoco.state import MjState

GROUPS = ("phase", "w", "amplitudes", "ha", "b")


class RealNaCPG:
    def __init__(self, n: int, dt: float, alpha: float, h: float, cf_margin: float, initial_xy) -> None:
        self.n, self.dt, self.alpha, self.h, self.cf_margin = n, dt, alpha, h, cf_margin
        self.mask = 1.0 - np.eye(n)
        self._xy0 = np.asarray(initial_xy, dtype=np.float64).reshape(n, 2)
        self.phase, self.w, self.amplitudes, self.ha, self.b = (np.zeros(n) for _ in GROUPS)
        self.reset()

    def set_flat_params(self, params) -> None:
        p = np.asarray(params, dtype=np.float64).reshape(len(GROUPS), self.n)
        self.phase, self.w, self.amplitudes, self.ha, self.b = (p[k].copy() for k in range(len(GROUPS)))
        self.reset()

    def reset(self) -> None:
        self.xy = self._xy0 + np.stack([np.zeros(self.n), self.b], axis=1)
        self.xy_dot = np.zeros((self.n, 2))

    def forward(self) -> np.ndarray:
        x, yc = self.xy[:, 0], self.xy[:, 1] - self.b
        r2 = x ** 2 + yc ** 2
        old_xd = self.xy_dot[:, 0]
        sign = np.where(old_xd > 0, 1.0, np.where(old_xd < 0, -1.0, 0.0))
        z = 1.0 - self.ha * sign
        rot = self.w / z
        grow = self.alpha * (1.0 - r2)
        own = np.stack([grow * x - rot * yc, rot * x + grow * yc], axis=1)
        dphi = self.phase[:, None] - self.phase[None, :]
        cos, sin = np.cos(dphi) * self.mask, np.sin(dphi) * self.mask
        coupling = self.h * np.stack([cos @ x - sin @ yc, sin @ x + cos @ yc], axis=1)
        xy_dot = own + coupling
        limit = (self.cf_margin * self.w ** 2 * self.dt)[:, None]
        xy_dot = np.clip(xy_dot, self.xy_dot - limit, self.xy_dot + limit)
        self.xy = self.xy + xy_dot * self.dt
        self.xy_dot = xy_dot
        return self.amplitudes * self.xy[:, 1]


class RealNaCPGController(Controller):
    @classmethod
    def name(cls): return "real_na_cpg"

    def __init__(self, weights, state: MjState, name: str, dt: float, alpha: float, h: float, cf_margin: float,
                 initial_xy):
        super().__init__(weights, state, name)
        self._mj_state = state
        self._cpg = RealNaCPG(self.hinges, dt, alpha, h, cf_margin, initial_xy)
        self.set_weights(weights)

    def reset(self, state: MjState):
        if self._mj_state != state:
            raise RuntimeError("Resetting with a different MjState. Actuators will not match")
        self._cpg.reset()

    @classmethod
    def num_parameters(cls, state: MjState, name: str, *args, **kwargs) -> int:
        return 5 * cls.get_num_joints(state, name)

    def set_weights(self, weights: np.ndarray):
        self._cpg.set_flat_params(weights)

    def extract_weights(self) -> np.ndarray:
        return np.concatenate([getattr(self._cpg, g) for g in GROUPS]).astype(np.float32)

    def __call__(self, state) -> None:
        angles = self._cpg.forward()
        for actuator, angle in zip(self._actuators, angles, strict=True):
            actuator.ctrl[:] = float(angle)
