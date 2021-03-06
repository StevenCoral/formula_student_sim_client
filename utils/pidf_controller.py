# Not based upon any imports

def filter_value(current_val, previous_val, alpha):
    return previous_val * alpha + current_val * (1.0-alpha)


def clamp(my_value, min_value, max_value):
    return max(min(my_value, max_value), min_value)


class PidfControl:
    def __init__(self, sample_time):
        self.kp = 1.0  # Proportional
        self.ki = 0.0  # Integral
        self.kd = 0.0  # Derivative
        self.kf = 0.0  # Feed-forward

        self.alpha = 0.5
        self.dt = sample_time

        self.pos_prev = 0.0
        self.vel_prev = 0.0
        self.filtered_vel = 0.0

        self.integral = 0.0
        self.max_integral = 1.0
        self.min_setpoint = 0.0

    def set_pidf(self, kp, ki, kd, kf):
        self.kp = kp  # Proportional
        self.ki = ki  # Integral
        self.kd = kd  # Derivative
        self.kf = kf  # Feed-forward

    def set_extrema(self, min_setpoint, max_integral):
        self.max_integral = max_integral
        self.min_setpoint = min_setpoint

    # The measurement input is a position, hence it is differentiated before computing the error.
    # Units depend on caller function's use case.
    def compute_velocity(self, pos, vel=None):
        # In most cases we would want to supply position data and derive velocity from it.
        # But, in some cases we might have the velocity already available,
        # for example if we have a gyro attached to the shaft.
        if vel is None:
            raw_vel = (pos - self.pos_prev) / self.dt
            self.pos_prev = pos
        else:
            raw_vel = vel

        self.filtered_vel = filter_value(raw_vel, self.vel_prev, self.alpha)
        if abs(self.filtered_vel) < 0.001:  # Disable unnecessary computations at low values
            self.filtered_vel = 0

        self.vel_prev = self.filtered_vel

    def compute_integral(self, err):
        self.integral += err * self.ki
        self.integral = clamp(self.integral, -self.max_integral, self.max_integral)  # Assuming symmetrical behaviour

    def velocity_control(self, setpoint, pos, vel=None):
        self.compute_velocity(pos, vel)
        err = setpoint - self.filtered_vel
        self.compute_integral(err)
        # We want to disable the integral term if the setpoint is too low to prevent currents:
        if abs(setpoint) < self.min_setpoint:
            self.integral = 0
        output = err * self.kp + self.integral + self.kf * setpoint
        return output

    def position_control(self, setpoint, pos, vel=None):
        self.compute_velocity(pos, vel)
        err = setpoint - pos
        self.compute_integral(err)
        output = err * self.kp + self.integral - self.kd * self.filtered_vel
        return output
