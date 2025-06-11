# Functional-Subnetworks-for-AARL

## 🧠  Outline

This project aims to develop a set of neural computation modules based on **Nengo**, designed for control systems running on the **Intel Loihi** neuromorphic chip. Our core contribution lies in implementing the **Functional Subnetwork (FSN)** method and adapting it to operate on **Loihi**, enabling fundamental arithmetic operations such as addition, multiplication, and differentiation. These are organized into a modular and composable computational subsystem. In this documentation, we present:

1. How the system is implemented
2. How to use the modules
3. The design of the testbench
4. Simulation results and performance

However, it is important to note that we cannot guarantee reliable performance of this network in actual robotic control scenarios. Due to the lack of a deployment platform, we have not tested these networks in real-world control applications. As the developer, I remain committed to supporting this work. If you are interested in applying these modules to new use cases, please feel free to contact me: ***kilobao377@gmail.com***

Additionally, the issue of **noise** must not be overlooked. Running the network on Loihi introduces significantly more noise compared to idealized simulations, and it remains uncertain whether such noise levels are tolerable in practical settings.

In any case, before you dive into our detailed design and implementation logic, we provide a quick setup guide below to help developers rapidly test and experiment with the modules. Please follow the instructions accordingly.

## Setup

Coming soon

---

## 1. Theoretical Realization of Arithmetic Functional Subnetworks

On neuromorphic computing platforms such as Intel’s Loihi chip, the fundamental computational unit is the **spiking neuron**, which represents neural activity through discrete events—namely, spike emissions. This is fundamentally different from traditional artificial neural networks based on continuous activation functions, such as ReLU or sigmoid units. As a result, continuous mathematical functions must be approximated through structured networks composed of spiking neurons.

To achieve this, we must systematically design network architectures that operate within spiking neural frameworks, such that the network’s dynamic behavior and decoded outputs closely approximate the desired target function. For instance, operations like addition, multiplication, or differentiation must be reformulated as structured connections and transformations between ensembles of spiking neurons.

This section introduces three basic arithmetic neural modules: the **Adder**, **Multiplier**, and **Differentiator**. Each of these modules is constructed based on explicit mathematical formulations, and their functionality is realized through network structure rather than training. These modules can be deployed on spiking neural platforms such as Loihi, serving as foundational components for more complex control and signal processing systems.

### Adder Subnetwork

The adder network is designed to approximate the function

$$
f(x, y) = x + y
$$

This is a linear operation, and in neural terms, it can be realized by simply summing the activity of two ensembles each encoding one input variable. The network comprises three ensembles: two input ensembles ***A*** and ***B***, and one downstream ensemble ***C*** which integrates the projections from both.

Mathematically, if $a(t)≈x(t)$ and $b(t)≈y(t)$ 

are the decoded outputs of ensembles ***A*** and ***B***, then we define connections to ***C*** such that:

$$
c(t)=w_a⋅a(t)+w_b⋅b(t)
$$

For a direct sum, we set $w_a​=w_b​=1$, The downstream ensemble ***C*** thus encodes the signal

$c(t)≈x(t)+y(t)$. The final output is decoded from ***C*** with a linear decoder.

This network is functionally exact within the linear range of the neural response curves and ensemble radius. Since the adder only requires linear summation, it represents the most straightforward of the FSNs and exhibits negligible approximation error under normal conditions.

### Multiplier Subnetwork

Multiplication is a nonlinear operation that cannot be realized by direct summation. Instead, the FSN multiplier approximates the function:

$$
f(x,y)=x⋅y
$$

To implement this without training, we exploit a known algebraic identity:

$$
x \cdot y = \frac{(x + y)^2 - (x - y)^2}{4}

$$

This allows us to reduce multiplication to a combination of addition, subtraction, and squaring—each of which can be implemented in a neural circuit.

The network structure comprises:

- A **sum node**: computes  $s_1=x+y$
- A **difference node**: computes  $s_2=x-y$
- Two ensembles $E_1, E_2$  approximate $s_1^2$ and $s_2^2$ using neural nonlinear tuning curves and a connection function $f(z) = z^2$
- A subtraction node: computes $s_3 = s_1^2 - s_2^2$
- A final scaling ensemble: computes $s_4 = \frac{1}{4}(s_3)$, yielding the approximate product $x \cdot y$

ach nonlinear transformation (square) is realized by decoding the square function over the ensemble’s response space:

$$

z_i(t) = \sum_j d_j \cdot a_j(t)

\text{where } d_j \text{ approximates } z^2 \text{ over the neuron's tuning curve}

$$

This approach leverages the fact that leaky integrate-and-fire (LIF) neurons exhibit nonlinear responses to current inputs, enabling them to represent nonlinear functions such as squares through appropriate decoders.

While this method introduces approximation errors—due to limitations in neural tuning range, dimensionality, and decoder resolution—it is effective and robust over the bounded domain defined by the ensemble radii.

### Differentiator Subnetwork

The differentiator FSN approximates the temporal derivative of a signal:

$$
f(t) = \frac{dx(t)}{dt}
$$

This is achieved using a **finite difference method**, where the derivative is approximated by:

$$
\frac{dx}{dt} \approx \frac{x(t) - x(t - \Delta t)}{\Delta t}

$$

In the neural implementation, the delay is introduced via a **low-pass filter** with time constant $\tau$, which approximates the value of the signal at a slightly earlier time:

- Let $x_{\text{curr}}(t)$ be the current value of the input ensemble
- Let $x_{\text{filt}}(t) \approx x(t - \tau)$ be the low-pass filtered signal:

$$
x_{\text{filt}}(t) = \frac{1}{\tau} \int_0^t e^{-(t-s)/\tau} x(s)\, ds
$$

Then, the approximate derivative is:

$$
\frac{dx}{dt} \approx \frac{x_{\text{curr}}(t) - x_{\text{filt}}(t)}{\tau}
$$

This difference is computed via two ensembles and decoded into a downstream ensemble. The final scaling by $1/\tau$  yields the desired derivative.

While this method does not provide an exact derivative—since filtering introduces smoothing and phase delay—it provides a biologically plausible, differentiable estimate over smooth signals, and is especially well-suited to signals where noise suppression is beneficial.

---

## 2. Implementation of Arithmetic Functional Subnetworks

In this section, we provide a detailed explanation of how the three arithmetic functional subnetworks—Adder, Multiplier, and Differentiator—are constructed based on the actual implementation code. All modules are encapsulated within the `ArithmeticFSN` class, implemented using the Nengo framework. The design ensures modularity and scalability, and the resulting networks are compatible with deployment on Intel's Loihi neuromorphic chip.

### Adder – Implementation and Code Explanation

The Adder module performs a linear summation of two input signals. The network comprises three ensembles: two encoding the individual inputs, and one integrating ensemble that produces the summed output.

```python
A = nengo.Ensemble(self.neuron_count, dimensions=1, ...)
B = nengo.Ensemble(self.neuron_count, dimensions=1, ...)
C = nengo.Ensemble(self.neuron_count, dimensions=1, ...)

nengo.Connection(input_a, A)
nengo.Connection(input_b, B)
nengo.Connection(A, C)
nengo.Connection(B, C)
```

Here, `A` and `B` receive the inputs, and `C` aggregates their activity through direct connections. The result is decoded at a downstream node:

```python
output = nengo.Node(size_in=1, label="Add_output")
nengo.Connection(C, output, synapse=0.01)
```

The use of `synapse=0.01` introduces mild low-pass filtering to smooth out the decoded signal. The entire structure is minimal yet precise, taking full advantage of LIF neuron linear integration.

### Multiplier – Implementation and Code Explanation

The Multiplier approximates the product of two inputs using the algebraic identity:

$$
x \cdot y = \frac{(x + y)^2 - (x - y)^2}{4}

$$

The implementation reflects this structure directly:

```python
sum_node = nengo.Ensemble(150, dimensions=1, ...)
diff_node = nengo.Ensemble(150, dimensions=1, ...)

nengo.Connection(input_a, sum_node)
nengo.Connection(input_b, sum_node)
nengo.Connection(input_a, diff_node)
nengo.Connection(input_b, diff_node, transform=-1)
```

Then, square both intermediate terms using ensemble-level function approximation:

```python
square_sum = nengo.Ensemble(200, dimensions=1, ...)
square_diff = nengo.Ensemble(200, dimensions=1, ...)

nengo.Connection(sum_node, square_sum, function=lambda x: x**2)
nengo.Connection(diff_node, square_diff, function=lambda x: x**2)
```

The squared values are subtracted and scaled:

```python
intermediate = nengo.Ensemble(150, dimensions=1, ...)
result = nengo.Ensemble(100, dimensions=1, ...)

nengo.Connection(square_sum, intermediate)
nengo.Connection(square_diff, intermediate, transform=-1)

nengo.Connection(intermediate, result, transform=0.25)
```

Finally, the product is decoded from the `result` ensemble:

```python
output = nengo.Node(size_in=1, label="Multiply_output")
nengo.Connection(result, output, synapse=0.01)
```

This structure demonstrates how algebraic transformations can be implemented as a cascade of ensembles and connections, making full use of neuron nonlinearity (squaring) and structural logic.

### Differentiator – Implementation and Code Explanation

The Differentiator estimates the derivative of a time-varying input signal using a low-pass filter and a difference computation.

First, the current input is encoded:

```python
input_ens = nengo.Ensemble(self.neuron_count, dimensions=1, ...)
nengo.Connection(input_signal, input_ens)
```

A delayed version is obtained via synaptic filtering:

```python
filtered = nengo.Ensemble(self.neuron_count, dimensions=1, ...)
nengo.Connection(input_ens, filtered, synapse=tau)
```

The difference is computed between the current and filtered signals:

```python
subtract = nengo.Ensemble(self.neuron_count, dimensions=1, ...)
nengo.Connection(input_ens, subtract)
nengo.Connection(filtered, subtract, transform=-1)
```

The result is scaled to approximate a derivative:

```python
output_ens = nengo.Ensemble(self.neuron_count, dimensions=1, ...)
nengo.Connection(subtract, output_ens, transform=1.0 / tau)
```

Finally, the output is decoded:

```python
output = nengo.Node(size_in=1, label="Differential_output")
nengo.Connection(output_ens, output, synapse=0.01)
```

This design aligns with the first-order finite difference approximation and uses biological mechanisms like filtering and suppression to implement differentiation in a spiking context.

---

### Unified Parameter Configuration

All submodules share a common global configuration defined in the `ArithmeticFSN` constructor. This includes:

```python
'neuron_type': nengo.LIF(),
'max_rates': Uniform(100, 200),
```

Module-specific layer configurations (such as radius values) are defined in `build_local_config()` as dictionaries indexed by module type. This setup ensures clarity and separation of concerns, enabling easy reconfiguration and reuse.

---

Please click the link below to view the testbench design and analysis of experimental results:

[testbench](Functional-Subnetworks-for-AARL%2020904b3f519f80aca631dec5f1466b29/testbench%2020904b3f519f805dac0af7b3188696a8.md)