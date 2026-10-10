"""Simple brushless DC motor + ESC model and operating-point solvers.

Motor model (standard 3-constant DC model, as used by QPROP / eCalc):
    Kt = 60 / (2 pi Kv)                [N m / A]   (Kv in rpm/V; Ke = Kt in SI)
    I  = Q / Kt + I0                   current needed to make shaft torque Q
    U  = omega / (Kv * 2pi/60) + I*Rm  motor terminal voltage needed at speed omega
    P_motor_in = U * I                 electrical power into the motor
    P_battery  = P_motor_in / eta_esc  (ESC losses lumped into one efficiency)

The ESC reduces the effective voltage to U = throttle * V_battery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

import bemt


@dataclass
class Motor:
    kv: float                  # [rpm / V]
    rm: float | None = None    # winding resistance [ohm]; None = estimate
    i0: float | None = None    # no-load current [A]; None = estimate
    esc_eff: float = 0.95

    @property
    def kt(self) -> float:
        return 60.0 / (2 * np.pi * self.kv)

    def with_estimates(self, voltage: float, power: float) -> "Motor":
        """Fill missing Rm / I0 with rough values for a typical hobby BLDC sized for `power` at `voltage`:
        copper loss ~6 % and no-load loss ~4 % of rated power (peak efficiency around 88-90 %)."""
        i_rated = power / voltage
        rm = self.rm if self.rm is not None else 0.06 * voltage ** 2 / power
        i0 = self.i0 if self.i0 is not None else 0.04 * i_rated
        return Motor(self.kv, rm, i0, self.esc_eff)

    def electrical(self, rpm: float, torque: float) -> dict:
        omega = rpm * 2 * np.pi / 60
        current = torque / self.kt + self.i0
        u = omega * self.kt + current * self.rm
        p_in = u * current
        p_shaft = torque * omega
        return dict(current=current, motor_voltage=u, motor_input_power=p_in,
                    battery_power=p_in / self.esc_eff,
                    motor_efficiency=p_shaft / p_in if p_in > 0 else 0.0,
                    drive_efficiency=p_shaft / (p_in / self.esc_eff) if p_in > 0 else 0.0)


def _rpm_bounds(motor: Motor, voltage: float):
    return 50.0, motor.kv * voltage * 0.9999


def solve_full_throttle(geom, motor: Motor, voltage: float, **bemt_kw):
    """RPM where the motor's torque at full throttle equals the prop's torque."""
    def resid(rpm):
        q = bemt.solve(geom, rpm, **bemt_kw).torque
        return motor.electrical(rpm, q)["motor_voltage"] - voltage
    lo, hi = _rpm_bounds(motor, voltage)
    if resid(lo) > 0:
        raise ValueError("Motor cannot even start this prop at this voltage (Rm or I0 too large?)")
    return brentq(resid, lo, hi, xtol=0.5)


def solve_for_power(geom, motor: Motor, power: float, rpm_max: float, **bemt_kw):
    """RPM at which the motor draws exactly `power` (motor input) driving this prop."""
    def resid(rpm):
        q = bemt.solve(geom, rpm, **bemt_kw).torque
        return motor.electrical(rpm, q)["motor_input_power"] - power
    lo = 50.0
    if resid(rpm_max) < 0:
        return None
    return brentq(resid, lo, rpm_max, xtol=0.5)


def solve_for_torque(geom, max_torque: float, rpm_max: float, **bemt_kw):
    """RPM (below rpm_max) at which the prop needs exactly `max_torque`; None if it never needs that much."""
    def resid(rpm):
        return bemt.solve(geom, rpm, **bemt_kw).torque - max_torque
    if resid(rpm_max) <= 0:
        return None
    lo = 50.0
    if resid(lo) > 0:
        raise ValueError(f"The prop needs more than the {max_torque:g} N m torque limit even at {lo:.0f} rpm")
    return brentq(resid, lo, rpm_max, xtol=0.5)


def motor_power_mode(geom, kv, power, voltage=None, rm=None, i0=None, esc_eff=0.95, max_torque=None,
                     **bemt_kw):
    """Operating point for a motor of given Kv and rated (max) input power.

    With a battery voltage: run full throttle; if that would draw more than the
    rated power, throttle back until the motor draws exactly the rated power.
    Without a voltage: find the battery voltage at which full throttle draws the
    rated power (useful to choose a cell count).
    max_torque [N m]: the motor's torque limit; the throttle comes back further if the prop would need more.
    """
    base = Motor(kv, rm, i0, esc_eff)
    if voltage is not None:
        motor = base.with_estimates(voltage, power)
        rpm_full = solve_full_throttle(geom, motor, voltage, **bemt_kw)
        res = bemt.solve(geom, rpm_full, **bemt_kw)
        el = motor.electrical(rpm_full, res.torque)
        limited_by = "battery voltage (full throttle)"
        if power and el["motor_input_power"] > power:
            rpm = solve_for_power(geom, motor, power, rpm_full, **bemt_kw)
            res = bemt.solve(geom, rpm, **bemt_kw)
            el = motor.electrical(rpm, res.torque)
            limited_by = "motor power rating (throttled back)"
        if max_torque and res.torque > max_torque:
            rpm = solve_for_torque(geom, max_torque, res.rpm, **bemt_kw)
            res = bemt.solve(geom, rpm, **bemt_kw)
            el = motor.electrical(rpm, res.torque)
            limited_by = "motor torque limit (throttled back)"
        el["throttle"] = el["motor_voltage"] / voltage
        el["battery_voltage"] = voltage
        el["battery_current"] = el["battery_power"] / voltage
        return res, el, motor, limited_by

    # No voltage given: solve for the voltage that makes full throttle = rated power
    def resid(v):
        m = base.with_estimates(v, power)
        try:
            rpm = solve_full_throttle(geom, m, v, **bemt_kw)
        except ValueError:
            return -power
        q = bemt.solve(geom, rpm, **bemt_kw).torque
        return m.electrical(rpm, q)["motor_input_power"] - power
    v_hi = 2.0
    while resid(v_hi) < 0:
        v_hi *= 2
        if v_hi > 2000:
            raise ValueError("Could not reach the rated power below 2000 V")
    v = brentq(resid, v_hi / 2 if v_hi > 2 else 0.5, v_hi, xtol=0.01)
    motor = base.with_estimates(v, power)
    rpm = solve_full_throttle(geom, motor, v, **bemt_kw)
    res = bemt.solve(geom, rpm, **bemt_kw)
    limited_by = "motor power rating (voltage solved for)"
    if max_torque and res.torque > max_torque:
        # the torque limit comes first: the voltage at which full throttle needs exactly max_torque
        # (Rm / I0 estimates stay sized on the rated power)
        def resid_q(u):
            try:
                return bemt.solve(geom, solve_full_throttle(geom, motor, u, **bemt_kw), **bemt_kw).torque \
                    - max_torque
            except ValueError:
                return -max_torque
        v = brentq(resid_q, 0.5, v, xtol=0.01)
        rpm = solve_full_throttle(geom, motor, v, **bemt_kw)
        res = bemt.solve(geom, rpm, **bemt_kw)
        limited_by = "motor torque limit (voltage solved for)"
    el = motor.electrical(rpm, res.torque)
    el["throttle"] = 1.0
    el["battery_voltage"] = v
    el["battery_current"] = el["battery_power"] / v
    return res, el, motor, limited_by
