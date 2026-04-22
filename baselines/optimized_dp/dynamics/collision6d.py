import heterocl as hcl
import math

# TwoVehicleCollision6D for optimized_dp (HeteroCL)
# State: [x1, y1, x2, y2, theta1, theta2]
# Each vehicle: x_dot = v*cos(theta), y_dot = v*sin(theta), theta_dot = u
# Controls: u1, u2 (angular velocities), both maximize (avoid collision)
# No disturbance
# Matches DeepReach TwoVehicleCollision6D class in dynamics/dynamics.py

class TwoVehicleCollision6D:
    def __init__(self, x=[0, 0, 0, 0, 0, 0], uMode="max", dMode="min"):
        self.x = x
        self.speed = 0.6
        self.wMax = 1.1
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        # Controls: u1 (omega for vehicle 1), u2 (omega for vehicle 2)
        # uMode=max (avoid): maximize Hamiltonian
        # H includes u1 * dV/dtheta1 + u2 * dV/dtheta2
        # => u_i = wMax * sign(dV/dtheta_i)
        opt_w1 = hcl.scalar(self.wMax, "opt_w1")
        opt_w2 = hcl.scalar(self.wMax, "opt_w2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        # Vehicle 1: control based on spat_deriv[4] (dV/dtheta1)
        with hcl.if_(spat_deriv[4] >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_w1[0] = -opt_w1[0]
        with hcl.elif_(spat_deriv[4] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w1[0] = -opt_w1[0]

        # Vehicle 2: control based on spat_deriv[5] (dV/dtheta2)
        with hcl.if_(spat_deriv[5] >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_w2[0] = -opt_w2[0]
        with hcl.elif_(spat_deriv[5] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w2[0] = -opt_w2[0]

        return (opt_w1[0], opt_w2[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        # No disturbance
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        # state: [x1, y1, x2, y2, theta1, theta2]
        x1_dot = hcl.scalar(0, "x1_dot")
        y1_dot = hcl.scalar(0, "y1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        y2_dot = hcl.scalar(0, "y2_dot")
        theta1_dot = hcl.scalar(0, "theta1_dot")
        theta2_dot = hcl.scalar(0, "theta2_dot")

        x1_dot[0] = self.speed * hcl.cos(state[4])
        y1_dot[0] = self.speed * hcl.sin(state[4])
        x2_dot[0] = self.speed * hcl.cos(state[5])
        y2_dot[0] = self.speed * hcl.sin(state[5])
        theta1_dot[0] = uOpt[0]
        theta2_dot[0] = uOpt[1]

        return (x1_dot[0], y1_dot[0], x2_dot[0], y2_dot[0], theta1_dot[0], theta2_dot[0])
