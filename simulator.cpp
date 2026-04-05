/**
 * 窜天猴（持续动力型）飞行轨迹模拟 - C++ 版本
 * 二维平面，考虑变质量、推力、重力和空气阻力（速度平方模型）
 * 发射角度通过终端输入（度）
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

// Physical parameters (SI units)
const double m0 = 0.015;        // initial total mass [kg]
const double mp = 0.004;        // propellant mass [kg]
const double tb = 1.0;          // burn time [s]
const double mu = mp / tb;      // mass flow rate [kg/s]
const double u_exhaust = 600.0; // exhaust velocity relative to rocket [m/s]
const double k = 3.46e-5;       // drag constant [kg/m]
const double g = 9.8;           // gravity acceleration [m/s²]

// State structure
struct State {
    double x, y, vx, vy, m;
};

// Compute derivatives (dx/dt)
State derivatives(const State& s, double t, double theta_rad) {
    double v = hypot(s.vx, s.vy);
    double dmdt = (t <= tb) ? -mu : 0.0;

    // Thrust direction: along instantaneous velocity (or initial angle when v ~ 0)
    double thrust_dir_x, thrust_dir_y;
    if (v < 1e-6) {
        thrust_dir_x = cos(theta_rad);
        thrust_dir_y = sin(theta_rad);
    } else {
        thrust_dir_x = s.vx / v;
        thrust_dir_y = s.vy / v;
    }

    double thrust_mag = u_exhaust * fabs(dmdt);
    double thrust_x = thrust_mag * thrust_dir_x;
    double thrust_y = thrust_mag * thrust_dir_y;

    double drag_x = -k * v * s.vx;
    double drag_y = -k * v * s.vy;

    double ax = (drag_x + thrust_x) / s.m;
    double ay = (drag_y + thrust_y) / s.m - g;

    return {s.vx, s.vy, ax, ay, dmdt};
}

// RK4 integrator
void rk4_step(State& s, double& t, double dt, double theta_rad) {
    State k1 = derivatives(s, t, theta_rad);
    State k2 = derivatives({s.x + 0.5*dt*k1.x, s.y + 0.5*dt*k1.y,
                            s.vx + 0.5*dt*k1.vx, s.vy + 0.5*dt*k1.vy,
                            s.m + 0.5*dt*k1.m}, t + 0.5*dt, theta_rad);
    State k3 = derivatives({s.x + 0.5*dt*k2.x, s.y + 0.5*dt*k2.y,
                            s.vx + 0.5*dt*k2.vx, s.vy + 0.5*dt*k2.vy,
                            s.m + 0.5*dt*k2.m}, t + 0.5*dt, theta_rad);
    State k4 = derivatives({s.x + dt*k3.x, s.y + dt*k3.y,
                            s.vx + dt*k3.vx, s.vy + dt*k3.vy,
                            s.m + dt*k3.m}, t + dt, theta_rad);

    s.x += dt * (k1.x + 2*k2.x + 2*k3.x + k4.x) / 6.0;
    s.y += dt * (k1.y + 2*k2.y + 2*k3.y + k4.y) / 6.0;
    s.vx += dt * (k1.vx + 2*k2.vx + 2*k3.vx + k4.vx) / 6.0;
    s.vy += dt * (k1.vy + 2*k2.vy + 2*k3.vy + k4.vy) / 6.0;
    s.m += dt * (k1.m + 2*k2.m + 2*k3.m + k4.m) / 6.0;
    t += dt;
}

int main() {
    // Input launch angle
    double theta_deg;
    cout << "Enter launch angle (degrees, 0-90): ";
    cin >> theta_deg;
    double theta_rad = theta_deg * acos(-1.0) / 180.0;

    // Initial state
    State s = {0.0, 0.0, 0.0, 0.0, m0};
    double t = 0.0;
    const double dt = 0.001;      // time step [s]
    const double t_max = 30.0;    // max simulation time [s]

    // Store trajectory for output (optional)
    vector<double> t_log, x_log, y_log, v_log;
    bool landed = false;
    double t_land = 0.0, x_land = 0.0;

    // Simulation loop
    while (t < t_max && !landed) {
        // Save data
        t_log.push_back(t);
        x_log.push_back(s.x);
        y_log.push_back(s.y);
        v_log.push_back(hypot(s.vx, s.vy));

        // Take a step
        double y_prev = s.y;
        rk4_step(s, t, dt, theta_rad);

        // Check for landing (y crosses zero from above)
        if (y_prev > 0.0 && s.y <= 0.0) {
            // Linear interpolation to find exact landing time and position
            double ratio = y_prev / (y_prev - s.y);
            t_land = t - dt + ratio * dt;
            x_land = (x_log.back() + ratio * (s.x - x_log.back()));
            landed = true;
        }
    }

    // Output results
    if (landed) {
        cout << "\nLaunch angle: " << theta_deg << "degrees" << endl;
        cout << "Landing time: " << t_land << " s" << endl;
        cout << "Landing horizontal distance: " << x_land << " m" << endl;
    } else {
        cout << "Not landed within simulation time." << endl;
    }

    // Write trajectory to CSV file for plotting (optional)
    ofstream out("trajectory.csv");
    if (out.is_open()) {
        out << "t,x,y,v\n";
        for (size_t i = 0; i < t_log.size(); ++i) {
            out << t_log[i] << "," << x_log[i] << "," << y_log[i] << "," << v_log[i] << "\n";
        }
        out.close();
        cout << "\nTrajectory data saved to trajectory.csv" << endl;
    }

    return 0;
}