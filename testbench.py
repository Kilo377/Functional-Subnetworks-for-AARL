# testbench.py
# This script contains test routines for the FSN (Functional Subnetwork)
# modules: adder, multiplier, and differentiator, implemented using Nengo
# and optionally deployed on Loihi.

import numpy as np
import matplotlib.pyplot as plt
import nengo
import nengo_loihi
from nengo.processes import Piecewise
from FSN import ArithmeticFSN

# ----------------------------------------------
# Global simulator switch: use Loihi if True
USE_LOIHI = True  # Set to False to use standard Nengo simulator
# ----------------------------------------------

def get_simulator(model, dt):
    """
    Return the appropriate simulator based on USE_LOIHI flag.
    If True, use Loihi backend; otherwise, use standard Nengo simulator.
    """
    if USE_LOIHI:
        return nengo_loihi.Simulator(model, dt=dt, precompute=True)
    else:
        return nengo.Simulator(model, dt=dt)

def test_single_module(op='add', T=5.0, dt=0.001):
    """
    Test a single FSN subnetwork: 'add', 'multiply', or 'differential'.

    Parameters:
    op : str
        Type of subnetwork to test ('add', 'multiply', 'differential').
    T : float
        Total simulation time in seconds.
    dt : float
        Simulation timestep.
    """
    fsn = ArithmeticFSN()

    if op == 'multiply':
        # Use piecewise constant inputs for multiplication test
        pwA = Piecewise({0: 1, 2.5: 10, 4: -10})
        pwB = Piecewise({0: 10, 1.5: 2, 3: 0, 4.5: 2})

        with nengo.Network(label="Test_MULTIPLY") as model:
            input_a = nengo.Node(pwA, label="inputA")
            input_b = nengo.Node(pwB, label="inputB")

            net, out = fsn.multiply(input_a, input_b)

            probe_inA = nengo.Probe(input_a)
            probe_inB = nengo.Probe(input_b)
            probe_out = nengo.Probe(out, synapse=0.01)

    else:
        # Use sine and constant signals for 'add' and 'differential'
        with nengo.Network(label=f"Test_{op.upper()}") as model:
            input_a = nengo.Node(lambda t: np.sin(2 * np.pi * t), label="input_a")
            input_b = nengo.Node(lambda t: 0.5, label="input_b")

            if op == 'add':
                net, out = fsn.add(input_a, input_b)
                expected = lambda t: np.sin(2 * np.pi * t) + 0.5
            elif op == 'differential':
                net, out = fsn.differential(input_a)
                expected = lambda t: 2 * np.pi * np.cos(2 * np.pi * t)
            else:
                raise ValueError("Parameter 'op' must be 'add', 'multiply' or 'differential'.")

            model.add(net)
            probe_out = nengo.Probe(out, synapse=0.01)

    sim = get_simulator(model, dt)
    with sim:
        sim.run(T)

    t = sim.trange()
    fsn_data = sim.data[probe_out]

    plt.figure(figsize=(8, 5))

    if op == 'multiply':
        inA_data = sim.data[probe_inA]
        inB_data = sim.data[probe_inB]
        expected_data = inA_data * inB_data

        plt.plot(t, fsn_data, label='FSN Output', linewidth=1.5)
        plt.plot(t, expected_data, '--', label='Expected (A×B)', color='C2', linewidth=1.5)
        plt.title("Test MULTIPLY FSN (Piecewise Constant Input)", fontsize=14)
        plt.ylim(-25, 40)

    else:
        plt.plot(t, fsn_data, label='FSN Output', linewidth=1.5)
        plt.plot(t, expected(t), '--', label='Expected', color='C2', linewidth=1.5)
        plt.title(f"Test {op.upper()} FSN", fontsize=14)
        plt.ylim(-7, 7) if op == 'differential' else plt.ylim(-1.5, 1.5)

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_add_then_mul(T=5.0, dt=0.001):
    """
    Test composite FSN: ADD followed by MUL.
    a = t, b = 2t → (a + b) * b = 6t^2
    """
    fsn = ArithmeticFSN()

    with nengo.Network(label="Test_ADD_mul") as model:
        input_a = nengo.Node(lambda t: t, label="input_a")
        input_b = nengo.Node(lambda t: 2 * t, label="input_b")

        net_add, out_add = fsn.add(input_a, input_b)
        net_mul, out_mul = fsn.multiply(out_add, input_b)

        model.add(net_add)
        model.add(net_mul)

        probe = nengo.Probe(out_mul, synapse=0.01)

    sim = get_simulator(model, dt)
    with sim:
        sim.run(T)

    t = sim.trange()
    fsn_data = sim.data[probe]

    plt.figure(figsize=(8, 5))
    plt.plot(t, fsn_data, label='FSN Output', linewidth=1.5)
    plt.plot(t, 6 * t**2, '--', label='Expected $6t^2$', color='C2', linewidth=1.5)
    plt.title("FSN Composition: ADD → MUL", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_mul_then_diff(T=5.0, dt=0.001):
    """
    Loihi-specific composite: Multiply followed by Differentiate.
    Optimized to approximate d(2t^2)/dt = 4t with minimal spiking noise.
    """
    fsn = ArithmeticFSN(neuron_count=100)

    # Adjust radii for stable high-value encoding
    fsn.config['multiply']['result']['radius'] = 120
    fsn.config['multiply']['intermediate']['radius'] = 400
    fsn.config['multiply']['square_sum']['radius'] = 400
    fsn.config['multiply']['square_diff']['radius'] = 400

    # Increase radii for derivative approximation
    fsn.config['differential']['input_ens']['radius'] = 20
    fsn.config['differential']['filtered']['radius'] = 20
    fsn.config['differential']['subtract']['radius'] = 20
    fsn.config['differential']['output_ens']['radius'] = 30

    tau = 0.1  # Smoothing parameter

    with nengo.Network(label="Test_MUL_diff_Loihi") as model:
        input_a = nengo.Node(lambda t: t, label="input_a")
        input_b = nengo.Node(lambda t: 2 * t, label="input_b")

        net_mul, out_mul = fsn.multiply(input_a, input_b)
        net_diff, out_diff = fsn.differential(out_mul, tau=tau)

        model.add(net_mul)
        model.add(net_diff)

        probe = nengo.Probe(out_diff, synapse=0.15)

    sim = get_simulator(model, dt)
    with sim:
        sim.run(T)

    t = sim.trange()
    fsn_data = sim.data[probe]

    plt.figure(figsize=(8, 5))
    plt.plot(t, fsn_data, label='FSN Output (Loihi)', linewidth=1.5)
    plt.plot(t, 4 * t, '--', label='Expected $4t$', color='C2', linewidth=1.5)
    plt.title("FSN Composition: MUL → DIFF (Loihi Tuning)", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.ylim(-1, 25)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Uncomment lines below to run desired tests
    test_single_module('add')
    test_single_module('multiply')
    test_single_module('differential')

    test_add_then_mul()
    test_mul_then_diff()


