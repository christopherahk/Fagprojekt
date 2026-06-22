## Mondrian Trees

The Mondrian forest is implemented across two header files, `FeatureExtractor.h` and `MondrianForest.h`, located on branch `model2` in the GitHub repository.

### Preprocessing

In the preprocessing step, we map raw spatiotemporal EMG windows $S \in \mathbb{R}^{N_{CH} \times W}$ into a compressed feature space $X \in \mathbb{R}^{N_{FEAT}}$, where $N_{CH}=56$ channels and $W=16$ timesteps. The resulting feature vector contains $N_{FEAT} = 56 \times (8 + 2) = 560$ inputs, partitioned into 8 temporal bins and 2 statistical features per channel.

The `FeatureExtractor.h` file implements the signal preprocessing pipeline. For each channel $c$, the raw time-domain signal segment $s_c(t)$, for $t=0, 1, ..., W-1$ is processed to capture the temporal distribution of muscle firing amplitudes. The segment is divided into 8 equal-width time bins. The bin width is given by:

$$
\frac{W}{B} = \frac{16}{8} = 2 \text{ samples}
$$

The integrated amplitude for bin $b$ is computed as the sum of absolute values, followed by log1p compression to produce the RBI features:

$$
X_{RBI}(c,b) = \ln\left(1 + \sum_{t=b\Delta b}^{(b+1)\Delta b-1}|s_c(t)|\right)
$$

To further capture the overall signal intensity and peaks within the window without relying on computationally expensive frequency-domain transformations, the mean and maximum amplitudes are computed for the entire channel segment and subsequently log1p compressed. Eliminating the initially proposed FFT module significantly reduced the computational complexity and SRAM footprint, allowing the saved memory to be reallocated towards increasing the depth of the Mondrian decision trees.

### Online Ensemble Architecture

The ensemble consists of $M = 15$ independent Mondrian Trees. It works deterministically within static memory constraints and uses data structs optimized for resource-restricted hardware.

To circumvent dynamic heap allocation and reduce the risk of memory fragmentation, the binary trees are stored within a static node pool array, with a strict max dimension of $N_{MAX} = 255$ nodes per tree.

Given an instance of a received training sample $(x,y)$, the system assigns a deterministic "random" seed to the linear congruential generator derived from the tree index $t$, and global sample count $N_{total}$.

The root bounding box ranges $(\mathbf{f}_{min}, \mathbf{f}_{max})$ are expanded to encompass $\mathbf{x}$. The instance then traverses down the tree recursively. At each node along the execution path, the structural weight indicators are updated [1]:

$$
n_{Samples} \leftarrow n_{Samples} + 1.0, \quad \text{classCounts}[y] \leftarrow \text{classCounts}[y] + 1.0
$$

If, while traversing, there is an encounter with an unsplit leaf node, a conditional Mondrian split is started. First, the total linear extension $\mathcal{E}$ of the input point outside the current known geometric boundaries is accumulated across all features:

$$
\mathcal{E} = \sum_{f=0}^{N_{FEAT}-1} \max(0, f_{min}^{(f)} - x^{(f)}) + \max(0, x^{(f)} - f_{max}^{(f)})
$$

If $\mathcal{E} > 10^{-6}$, a spatial split time increment $\Delta \tau$ is drawn from an exponential distribution parameterized by the rate $\mathcal{E}$:

$$
\Delta \tau = \frac{-\ln(u)}{\mathcal{E}}, \quad u \sim \mathcal{U}(10^{-7}, 1.0)
$$

The conditional split evaluates the budget threshold criteria:

$$
\tau_{new} = \tau_{parent} + \Delta \tau \le \lambda \quad (\text{where } \lambda = 6.0)
$$

If $\tau_{new} \le \lambda$ and the node pool has capacity left ($nNodes + 2 \le 255$), a new structural split is introduced. The target split feature dimension $f_{split}$ is chosen randomly with a probability proportional to its spatial extension contribution. The exact numerical split threshold is sampled uniformly within that selected feature's extension boundary. The current leaf is then transformed into an internal node, allocating two child nodes and distributing historical class count distributions appropriately based on the geometric side of the new plane.

For class probability estimation, the vector $\mathbf{x}$ travels down the internal decisions of tree $t$ until it settles in a designated leaf node. To prevent absolute zero probabilities for classes unobserved along that pathway, Laplace smoothing with parameter $\alpha = 1.0$ is applied:

$$
P_t(Y = c \mid \mathbf{x}) = \frac{\text{classCounts}[c] + \alpha}{n_{Samples} + \alpha \cdot C}
$$

where $C = 3$ classes. The final ensemble probability distribution is the uniform average over all trees:

$$
P(Y = c \mid \mathbf{x}) = \frac{1}{M} \sum_{t=0}^{M-1} P_t(Y = c \mid \mathbf{x})
$$
