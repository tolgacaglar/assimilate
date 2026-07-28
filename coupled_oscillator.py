"""
Coupled harmonic oscillator simulation.

Physical setup:
    wall --[k1, c1]-- mass1 --[k2, c2]-- mass2 --(external force F(t))

Only one wall is present. mass1 is attached to that wall by a spring (k1)
with a damper in parallel (c1). mass1 and mass2 are connected to each other
by a coupling spring (k2), also with a damper in parallel (c2). mass2 has no
connection to a second wall; instead it is driven by an external, periodic
square-like force F(t). Each cycle's period and amplitude are randomly
jittered a little, so no two cycles are exactly alike, and each sign switch
is smoothed into a gradual (tanh-shaped) transition instead of an instant
jump, making F(t) continuous and differentiable everywhere.

Each damper resists the relative velocity across its spring: the wall
spring's damper resists mass1's velocity (the wall doesn't move), while the
coupling spring's damper resists the relative velocity between mass1 and
mass2.

Equations of motion (Newton's second law, F = m*a):
    m1 * x1'' = -k1*x1 - c1*x1' - k2*(x1 - x2) - c2*(x1' - x2')
    m2 * x2'' = -k2*(x2 - x1) - c2*(x2' - x1') + F(t)

The script integrates these equations forward in time using
scipy.integrate.solve_ivp and plots the resulting positions of both masses.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Parameters (edit these to change the simulation)
# ---------------------------------------------------------------------------

# Masses [kg]
mass1 = 1.0
mass2 = 1.0

# Spring constants [N/m]
k1 = 4.0   # spring connecting mass1 to the wall
k2 = 1.0    # coupling spring connecting mass1 and mass2

# Damping coefficients [N*s/m], one damper per spring, in parallel with it
# (set to 0 for no damping / undamped motion)
c1 = 0.5    # damper in parallel with the wall spring (k1)
c2 = 0.5    # damper in parallel with the coupling spring (k2)

# External force on mass2 [N]: a periodic step-wise (square-like) force that
# alternates between +amplitude and -amplitude every half period. Each cycle's
# period and amplitude are randomly jittered by a small fraction, so the wave
# still looks like a square wave but is never perfectly regular.
force_amplitude = 1.0          # base magnitude of the applied force [N]
force_amplitude_jitter = 0.15  # max random amplitude variation, as a fraction of force_amplitude (0 = none)
force_period = 10              # base time for one full square-wave cycle [s]
force_period_jitter = 0.15     # max random period variation, as a fraction of force_period (0 = none)
force_random_seed = 42         # seed for reproducible randomness
force_transition_time = 0.1    # time constant [s] controlling how gradual each sign switch is
                                # (larger = slower, more rounded-off drops; smaller = closer to an abrupt jump)

# Initial conditions
x1_initial = 1.0   # initial displacement of mass1 from its equilibrium [m]
v1_initial = 0.0   # initial velocity of mass1 [m/s]
x2_initial = 0.0   # initial displacement of mass2 from its equilibrium [m]
v2_initial = 0.0   # initial velocity of mass2 [m/s]

# Equilibrium positions [m], measured from the wall. Used only to convert
# each mass's displacement (x1, x2) into an absolute position for plotting.
mass1_equilibrium_position = 2.0   # where mass1 sits at rest, from the wall
mass2_equilibrium_position = 5.0   # where mass2 sits at rest, from the wall

# Simulation time settings
start_time = 0.0     # simulation start time [s]
end_time = 50.0      # simulation end time [s]
num_time_points = 10000  # number of points at which to evaluate the solution


# ---------------------------------------------------------------------------
# Build a square-like force with per-cycle random variation
# ---------------------------------------------------------------------------

# Generate the times at which the force switches sign (every half period) and
# the amplitude held until each switch. Each cycle's period and amplitude are
# independently jittered by a random fraction, so the wave keeps its square
# shape but drifts slightly from one cycle to the next.
_force_rng = np.random.default_rng(force_random_seed)

_switch_times = [start_time]     # boundaries between constant-force segments
_segment_amplitudes = []         # force value held during each segment

_current_time = start_time
while _current_time < end_time:
    _cycle_period = force_period * (1 + _force_rng.uniform(-force_period_jitter, force_period_jitter))
    _cycle_amplitude = force_amplitude * (1 + _force_rng.uniform(-force_amplitude_jitter, force_amplitude_jitter))
    _half_period = _cycle_period / 2

    _segment_amplitudes.append(_cycle_amplitude)     # first half of the cycle: +amplitude
    _switch_times.append(_current_time + _half_period)

    _segment_amplitudes.append(-_cycle_amplitude)    # second half of the cycle: -amplitude
    _switch_times.append(_current_time + _cycle_period)

    _current_time += _cycle_period

_switch_times = np.array(_switch_times)
_segment_amplitudes = np.array(_segment_amplitudes)

# Recast the step-wise waveform as a smooth sum of tanh transitions: starting
# from the first segment's level, add each subsequent level change as a
# gradual tanh ramp centered on its switch time instead of an instant jump.
# tanh is continuous and differentiable everywhere, so the resulting force is
# too, while still looking like a (rounded-off) square wave.
_base_level = _segment_amplitudes[0]
_transition_times = _switch_times[1:-1]     # switch times between segments (excludes the start and the trailing end)
_level_changes = np.diff(_segment_amplitudes)  # how much the level changes at each transition


def external_force_on_mass2(t):
    """
    Smoothed, square-like force applied to mass2.

    Same jittered period/amplitude levels as a square wave, but each sign
    switch is a gradual tanh ramp (width set by force_transition_time)
    rather than an instant jump, so the force is continuous and
    differentiable everywhere.
    """
    t_array = np.atleast_1d(np.asarray(t, dtype=float))
    # Each transition contributes level_change * 0.5*(1 + tanh((t - transition_time) / force_transition_time)),
    # which smoothly rises from 0 (well before the transition) to level_change (well after it).
    ramps = 0.5 * (1 + np.tanh((t_array[:, None] - _transition_times[None, :]) / force_transition_time))
    force_values = _base_level + ramps @ _level_changes
    return force_values.item() if np.isscalar(t) else force_values


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def coupled_oscillator_equations(t, state):
    """
    Compute the time derivative of the system state.

    The state vector is [x1, v1, x2, v2], where x1/x2 are the mass
    positions and v1/v2 are their velocities. This function returns
    [v1, a1, v2, a2], i.e. the derivative of each state variable, which is
    what solve_ivp needs to advance the solution forward in time.
    """
    x1, v1, x2, v2 = state

    # Relative displacement and velocity across the coupling spring, used by
    # both its spring force and its damper force.
    coupling_displacement = x1 - x2
    coupling_velocity = v1 - v2

    # Net force on each mass is the sum of its spring force(s), damper
    # force(s), and any external force acting on it.
    force1 = -k1 * x1 - c1 * v1 - k2 * coupling_displacement - c2 * coupling_velocity
    force2 = k2 * coupling_displacement + c2 * coupling_velocity + external_force_on_mass2(t)

    # Acceleration = force / mass (Newton's second law).
    a1 = force1 / mass1
    a2 = force2 / mass2

    return [v1, a1, v2, a2]


# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------

initial_state = [x1_initial, v1_initial, x2_initial, v2_initial]
time_points = np.linspace(start_time, end_time, num_time_points)

solution = solve_ivp(
    fun=coupled_oscillator_equations,
    t_span=(start_time, end_time),
    y0=initial_state,
    t_eval=time_points,
    method="RK45",  # adaptive-step Runge-Kutta method
    max_step=min(force_period * (1 - force_period_jitter), force_transition_time) / 20,  # resolve the shortest cycle/transition
    rtol=1e-6,      # relative tolerance for adaptive step size
    atol=1e-9       # absolute tolerance for adaptive step size
)

time = solution.t
x1 = solution.y[0]
v1 = solution.y[1]
x2 = solution.y[2]
v2 = solution.y[3]
applied_force = external_force_on_mass2(time)

# Absolute positions of each mass (equilibrium position + displacement),
# used for plotting instead of raw displacement.
position1 = mass1_equilibrium_position + x1
position2 = mass2_equilibrium_position + x2


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------

figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

axes[0].plot(time, position1, label="mass1 (position1)")
axes[0].plot(time, position2, label="mass2 (position2)")
axes[0].set_ylabel("Position [m]")
axes[0].set_title("Coupled Harmonic Oscillator: Position vs. Time")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, v1, label="mass1 (v1)")
axes[1].plot(time, v2, label="mass2 (v2)")
axes[1].set_ylabel("Velocity [m/s]")
axes[1].set_title("Coupled Harmonic Oscillator: Velocity vs. Time")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(time, applied_force, label="F(t) on mass2", color="tab:red")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Force [N]")
axes[2].set_title("External Smoothed Square-like Force on mass2")
axes[2].legend()
axes[2].grid(True)

figure.tight_layout()
plt.show()
