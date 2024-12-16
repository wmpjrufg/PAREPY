---
layout: home
parent: distributions
grand_parent: Framework
nav_order: 1
has_children: true
has_toc: true
title: uniform_sampling
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<p align="justify">
    This function generates a Uniform sampling between \(a\) (minimum) and \(b\) (maximum).
</p>

```python
u = uniform_sampling(parameters, method, n_samples, seed)
```

Input variables
{: .label .label-yellow }

<table style="width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tr>
        <td><code>parameters</code></td>
        <td>
            <p align="justify">
            Dictionary of parameters for uniform distribution. Keys:
            <ul>
                <li><code>'min'</code>: Minimum value of the uniform distribution [float]</li>
                <li><code>'max'</code>: Maximum value of the uniform distribution [float]</li>
            </ul>
            </p>
        </td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>method</code></td>
        <td>
            <p align="justify">Sampling method. Supports the following values:
            <ul>
                <li><code>'mcs'</code>: Crude Monte Carlo Sampling</li>
                <li><code>'lhs'</code>: Latin Hypercube Sampling</li>
            </ul>
            </p>
        </td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>n_samples</code></td>
        <td>Number of samples</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>seed</code></td>
        <td>Seed for random number generation. Use <code>None</code> for a random seed</td>
        <td>Integer or None</td>
    </tr>
</table>

Output variables
{: .label .label-yellow }

<table style="width:100%">
   <thead>
     <tr>
       <th>Name</th>
       <th>Description</th>
       <th>Type</th>
     </tr>
   </thead>
   <tr>
       <td><code>u</code></td>
       <td>Random samples</td>
       <td>List</td>
   </tr>
</table>

Example 1
{: .label .label-blue }

<p align="justify">
    <i>
        In this example, we will use the <code>uniform_sampling</code> function from the <code>parepy_toolbox</code> to generate two random samples (\(n=400\)) following a uniform distribution. The first set is sampled using the Monte Carlo Sampling (MCS) method, and the second using the Latin Hypercube Sampling (LHS) method. The range for both distributions is defined as \([10, 20]\). The results are visualized using histograms with Kernel Density Estimates (KDE) plotted (using matplotlib lib) side-by-side for comparison.
    </i>
</p>

```python
# Libraries
import matplotlib.pyplot as plt

from parepy_toolbox import uniform_sampling

# Sampling
n = 400
x = uniform_sampling({'min': 10, 'max': 20}, 'mcs', n)
y = uniform_sampling({'min': 10, 'max': 20}, 'lhs', n)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.histplot(x, kde=True, bins=30, color='blue', ax=axes[0], alpha=0.6, edgecolor='black')
axes[0].set_title('MCS Sampling')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Density')
sns.histplot(y, kde=True, bins=30, color='green', ax=axes[1], alpha=0.6, edgecolor='black')
axes[1].set_title('LHS Sampling')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Density')
plt.tight_layout()
plt.show()
```

<center>
    <img src="assets/images/uniform_sampling.png" height="auto">
    <p align="center"><b>Figure 1.</b> Uniform variable example.</p>
</center>

Example 2
{: .label .label-blue }

<p align="justify">
    <i>
    In this example, we will use the <code>uniform_sampling</code> function from the <code>parepy_toolbox</code> to generate two random samples (\(n=3\)) following a uniform distribution. Using the Monte Carlo algorithm and the specific seed (<code>seed=25</code>), we generated 3 times and compared the results.
    </i>
</p>

```python
# Library
from parepy_toolbox import uniform_sampling

# Sampling
n = 3
x0 = uniform_sampling({'min': 10, 'max': 20}, 'mcs', n, 25)
x1 = uniform_sampling({'min': 10, 'max': 20}, 'mcs', n, 25)
x2 = uniform_sampling({'min': 10, 'max': 20}, 'mcs', n, 25)
print(x0, '\n', x1, '\n', x2)
```

<p align = "justify">
    <i>Output details.</i>
</p>

```bash
[11.607212332320078, 15.003120351710036, 12.16598464462817] 
[11.607212332320078, 15.003120351710036, 12.16598464462817] 
[11.607212332320078, 15.003120351710036, 12.16598464462817]
```

{: .important }
> Note that using the seed 25 by 3 times, we can generate the same values in a random variable.