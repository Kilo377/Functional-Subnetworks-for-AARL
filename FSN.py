import nengo
import nengo_loihi
from nengo.dists import Uniform

class ArithmeticFSN:
    def __init__(self, neuron_count=100, global_config=None):
        """
        Initialize the ArithmeticFSN class, which provides functional subnetworks
        for arithmetic operations (addition, multiplication, differentiation)
        implemented using spiking neural models in Nengo.

        Parameters
        ----------
        neuron_count : int
            Number of neurons per ensemble used in the subnetwork layers.
        global_config : dict or None
            Dictionary specifying global neuron parameters shared across layers,
            such as neuron_type and max_rates. If None, default values will be used.
        """
        self.neuron_count = neuron_count
        self.global_config = global_config or self.default_global_config()
        self.config = self.build_local_config()

    def default_global_config(self):
        """
        Define default configuration shared across all ensembles unless overridden.
        Returns
        -------
        dict
            Dictionary containing:
              - neuron_type: LIF (Leaky Integrate-and-Fire) neurons
              - max_rates: Spiking rate distribution (Uniform between 100 and 200 Hz)
        """
        return {
            'neuron_type': nengo.LIF(),           # Neuron model must be instantiated
            'max_rates': Uniform(100, 200),       # Rate range for encoding dynamics
        }

    def build_local_config(self):
        """
        Define per-module local configuration including ensemble radius values
        for each stage in the network. These radii determine value encoding ranges.

        Returns
        -------
        dict
            Layer-specific radius configuration for each operation.
        """
        return {
            'add': {
                'layers': [
                    {'radius': 10},  # Input A
                    {'radius': 10},  # Input B
                    {'radius': 20}   # Output: A + B
                ]
            },
            'multiply': {
                # Radii configured to support large intermediate values
                'sum_node': {'radius': 20},
                'diff_node': {'radius': 20},
                'square_sum': {'radius': 400},
                'square_diff': {'radius': 400},
                'intermediate': {'radius': 500},
                'result': {'radius': 100}
            },
            'differential': {
                # Radius scaling chosen for small input slopes and filtered signals
                'input_ens': {'radius': 2},
                'filtered': {'radius': 2},
                'subtract': {'radius': 2},
                'output_ens': {'radius': 10}
            }
        }

    def add(self, input_a, input_b):
        """
        Construct a spiking neural network to perform addition:
        output = input_a + input_b

        Parameters
        ----------
        input_a, input_b : nengo.Node or nengo.Ensemble
            Inputs to be summed.

        Returns
        -------
        addition_net : nengo.Network
            The constructed network performing addition.
        output : nengo.Node
            Output node carrying the decoded sum.
        """
        cfg = self.config['add']['layers']
        addition_net = nengo.Network(label="Add")
        with addition_net:
            # Input ensembles A and B
            A = nengo.Ensemble(self.neuron_count, 1,
                               neuron_type=self.global_config['neuron_type'],
                               max_rates=self.global_config['max_rates'],
                               radius=cfg[0]['radius'])
            B = nengo.Ensemble(self.neuron_count, 1,
                               neuron_type=self.global_config['neuron_type'],
                               max_rates=self.global_config['max_rates'],
                               radius=cfg[1]['radius'])
            # Output ensemble C accumulates both inputs
            C = nengo.Ensemble(self.neuron_count, 1,
                               neuron_type=self.global_config['neuron_type'],
                               max_rates=self.global_config['max_rates'],
                               radius=cfg[2]['radius'])

            # Connect input sources to respective ensembles
            nengo.Connection(input_a, A)
            nengo.Connection(input_b, B)
            # Linearly sum inputs into output ensemble
            nengo.Connection(A, C)
            nengo.Connection(B, C)

            # Output node receives decoded signal from ensemble C
            output = nengo.Node(size_in=1, label="Add_output")
            nengo.Connection(C, output, synapse=0.01)

        return addition_net, output

    def multiply(self, input_a, input_b):
        """
        Construct a network to compute element-wise multiplication using the identity:
        (x + y)^2 - (x - y)^2 = 4xy

        Parameters
        ----------
        input_a, input_b : nengo.Node or nengo.Ensemble
            Inputs to be multiplied.

        Returns
        -------
        mult_net : nengo.Network
            The multiplication network.
        output : nengo.Node
            The decoded product output.
        """
        cfg = self.config['multiply']
        mult_net = nengo.Network(label="Multiply")
        with mult_net:
            # First stage: compute x + y and x - y
            sum_node = nengo.Ensemble(150, 1,
                                       neuron_type=self.global_config['neuron_type'],
                                       max_rates=self.global_config['max_rates'],
                                       radius=cfg['sum_node']['radius'])
            diff_node = nengo.Ensemble(150, 1,
                                        neuron_type=self.global_config['neuron_type'],
                                        max_rates=self.global_config['max_rates'],
                                        radius=cfg['diff_node']['radius'])

            # Square the outputs (nonlinear transformation)
            square_sum = nengo.Ensemble(200, 1,
                                        neuron_type=self.global_config['neuron_type'],
                                        max_rates=self.global_config['max_rates'],
                                        radius=cfg['square_sum']['radius'])
            square_diff = nengo.Ensemble(200, 1,
                                         neuron_type=self.global_config['neuron_type'],
                                         max_rates=self.global_config['max_rates'],
                                         radius=cfg['square_diff']['radius'])

            # Compute (square_sum - square_diff)
            intermediate = nengo.Ensemble(150, 1,
                                          neuron_type=self.global_config['neuron_type'],
                                          max_rates=self.global_config['max_rates'],
                                          radius=cfg['intermediate']['radius'])

            # Final result after scaling by 0.25 (divide by 4)
            result = nengo.Ensemble(100, 1,
                                    neuron_type=self.global_config['neuron_type'],
                                    max_rates=self.global_config['max_rates'],
                                    radius=cfg['result']['radius'])

            # Input connections
            nengo.Connection(input_a, sum_node)
            nengo.Connection(input_b, sum_node)
            nengo.Connection(input_a, diff_node)
            nengo.Connection(input_b, diff_node, transform=-1)

            # Nonlinear transformation: square
            nengo.Connection(sum_node, square_sum, function=lambda x: x**2)
            nengo.Connection(diff_node, square_diff, function=lambda x: x**2)

            # Subtract square_diff from square_sum
            nengo.Connection(square_sum, intermediate)
            nengo.Connection(square_diff, intermediate, transform=-1)

            # Scale down to get final product
            nengo.Connection(intermediate, result, transform=0.25)

            # Output node with low-pass filtering
            output = nengo.Node(size_in=1, label="Multiply_output")
            nengo.Connection(result, output, synapse=0.01)

        return mult_net, output

    def differential(self, input_signal, tau=0.1):
        """
        Construct a network to approximate the time derivative of an input signal
        using a low-pass filtered delay and subtraction mechanism.

        Parameters
        ----------
        input_signal : nengo.Node or nengo.Ensemble
            Signal to differentiate.
        tau : float
            Synaptic time constant for low-pass filtering (delay window).

        Returns
        -------
        diff_net : nengo.Network
            The differentiator network.
        output : nengo.Node
            The decoded derivative output.
        """
        cfg = self.config['differential']
        diff_net = nengo.Network(label="Differential")
        with diff_net:
            # Current input
            input_ens = nengo.Ensemble(self.neuron_count, 1,
                                       neuron_type=self.global_config['neuron_type'],
                                       max_rates=self.global_config['max_rates'],
                                       radius=cfg['input_ens']['radius'])

            # Delayed version via synapse (low-pass filter)
            filtered = nengo.Ensemble(self.neuron_count, 1,
                                      neuron_type=self.global_config['neuron_type'],
                                      max_rates=self.global_config['max_rates'],
                                      radius=cfg['filtered']['radius'])

            # Subtraction node: input - delayed_input
            subtract = nengo.Ensemble(self.neuron_count, 1,
                                      neuron_type=self.global_config['neuron_type'],
                                      max_rates=self.global_config['max_rates'],
                                      radius=cfg['subtract']['radius'])

            # Output ensemble (scaled difference)
            output_ens = nengo.Ensemble(self.neuron_count, 1,
                                        neuron_type=self.global_config['neuron_type'],
                                        max_rates=self.global_config['max_rates'],
                                        radius=cfg['output_ens']['radius'])

            # Network connections
            nengo.Connection(input_signal, input_ens)
            nengo.Connection(input_ens, filtered, synapse=tau)
            nengo.Connection(input_ens, subtract)
            nengo.Connection(filtered, subtract, transform=-1)

            # Scale difference by 1/tau to approximate derivative
            nengo.Connection(subtract, output_ens, transform=1.0 / tau)

            # Final output node
            output = nengo.Node(size_in=1, label="Differential_output")
            nengo.Connection(output_ens, output, synapse=0.01)

        return diff_net, output

