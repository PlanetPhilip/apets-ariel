import numpy as np
import torch

from ariel.simulation.controllers.na_cpg import E, NaCPG
from .abstract import Controller
from ..mujoco.state import MjState

_CLAMP_WIDTH = 10.0  # matches na_cpg.py's hard-coded `diff = 10`


class _FastNaCPG(NaCPG):
    """Vectorized NaCPG.forward() — ported from hunger_games/src/controllers/na_cpg_fast.py
    so weights trained there reproduce the exact same trajectory here. Uses synchronous
    (Jacobi-style) oscillator coupling instead of na_cpg.py's sequential one; the two only
    diverge at second order in dt, but that's the variant the archived weights were tuned
    against, so it's kept exact rather than "fixed" to match upstream."""

    def __init__(self, adjacency_dict: dict[int, list[int]], *args, **kwargs) -> None:
        super().__init__(adjacency_dict, *args, **kwargs)
        mask = torch.zeros(self.n, self.n)
        for i, neighbours in adjacency_dict.items():
            for j in neighbours:
                mask[i, j] = 1.0
        self.register_buffer("_adj_mask", mask)

    def forward(self, time: float | None = None) -> torch.Tensor:
        if time is not None and abs(float(time)) < 1e-12:
            self.reset()

        with torch.inference_mode():
            xy = self.xy
            x_dot_old = self.xy_dot_old[:, 0]

            r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
            term_a = self.alpha * (1 - r2 ** 2)
            zeta = 1 - self.ha * ((x_dot_old + E) / (torch.abs(x_dot_old) + E))
            term_b = self.w / (zeta + E)
            k_matrix = torch.stack([
                torch.stack([term_a, -term_b], dim=-1),
                torch.stack([term_b, term_a], dim=-1),
            ], dim=-2)
            term_a_vec = torch.bmm(k_matrix, xy.unsqueeze(-1)).squeeze(-1)

            dphi = self.phase.unsqueeze(1) - self.phase.unsqueeze(0)
            cos_d, sin_d = torch.cos(dphi), torch.sin(dphi)
            r_matrix = torch.stack([
                torch.stack([cos_d, -sin_d], dim=-1),
                torch.stack([sin_d, cos_d], dim=-1),
            ], dim=-2)
            r_matrix = r_matrix * self._adj_mask.view(self.n, self.n, 1, 1)
            term_b_vec = torch.einsum("nmij,mj->ni", r_matrix, xy)

            combined_dot = term_a_vec + term_b_vec
            combined_dot = torch.clamp(
                combined_dot, self.xy_dot_old - _CLAMP_WIDTH, self.xy_dot_old + _CLAMP_WIDTH
            )

            xy_new = xy + combined_dot * self.dt
            self.xy_dot_old = xy.clone()
            self.xy = xy_new
            angles = self.amplitudes * xy_new[:, 1]

            if self.hard_bounds is not None:
                angles = torch.clamp(angles, min=self.hard_bounds[0], max=self.hard_bounds[1])

        if torch.isnan(angles).any():
            raise ValueError(f"NaN values detected in the angle signal.\n{angles=}")

        self.angles = angles
        return self.angles.clone()


class NaCPGController(Controller):
    """Hardware-side port of hunger_games' na_cpg controller (a fully-connected NA-CPG,
    see ariel.simulation.controllers.na_cpg.NaCPG). `dt`, `initial_xy` and
    `initial_xy_dot_old` must match the archive built by export_hardware_archive.py:
    xy/xy_dot_old are random-but-seeded, non-learnable buffers (not part of `weights`),
    so reproducing them from a bare seed here would need this file's torch/RNG behavior
    to exactly match the exporter's — baking the actual (already-permuted) values into
    the archive avoids that dependency."""

    @classmethod
    def name(cls): return "na_cpg"

    def __init__(self, weights, state: MjState, name: str, dt: float,
                 initial_xy, initial_xy_dot_old):
        super().__init__(weights, state, name)
        self._mj_state = state
        adjacency = {i: [j for j in range(self.hinges) if j != i] for i in range(self.hinges)}
        self._cpg = _FastNaCPG(adjacency, dt=dt)
        self.set_weights(weights)

        xy = torch.tensor(initial_xy, dtype=torch.float32)
        xy_dot_old = torch.tensor(initial_xy_dot_old, dtype=torch.float32)
        self._cpg.xy.data = xy
        self._cpg.xy_dot_old.data = xy_dot_old
        self._cpg.initial_state["xy"] = xy.clone()
        self._cpg.initial_state["xy_dot_old"] = xy_dot_old.clone()

    def reset(self, state: MjState):
        if self._mj_state != state:
            raise RuntimeError(f"Resetting with a different MjState. Actuators will not match"
                               f" ({id(self._mj_state)} != {id(state)})")
        self._cpg.reset()

    @classmethod
    def num_parameters(cls, state: MjState, name: str, *args, **kwargs) -> int:
        return 5 * cls.get_num_joints(state, name)

    def set_weights(self, weights: np.ndarray):
        self._cpg.set_flat_params(torch.as_tensor(np.asarray(weights, dtype=np.float32)))

    def extract_weights(self) -> np.ndarray:
        return self._cpg.get_flat_params().detach().cpu().numpy()

    def __call__(self, state) -> None:
        angles = self._cpg.forward(time=float(state.time)).detach().cpu().numpy()
        for actuator, angle in zip(self._actuators, angles, strict=True):
            actuator.ctrl[:] = float(angle)
