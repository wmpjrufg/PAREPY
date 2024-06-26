---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 1
has_children: false
has_toc: false
title: sampling
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>sampling</h3>
<br>
<p align = "justify">
    This algorithm generates a set of random numbers according to a type of distribution.
</p>

```python
random_sampling = sampling(n_samples, d, model, variables_setup)
```

Input variables
{: .label .label-yellow }

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tr>
        <td><code>n_samples</code></td>
        <td>Number of samples</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>d</code></td>
        <td>Number of dimensions</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>model</code></td>
        <td><a href="#mode">Model parameters</a></td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>variables_setup</code></td>
        <td><p align = "justify">Random variable parameters (list of dictionary format). <a href="#vars">See the example.</a></p></td>
        <td>List</td>
    </tr>
</table>

Output variables
{: .label .label-yellow}

<table style = "width:100%">
   <thead>
     <tr>
       <th>Name</th>
       <th>Description</th>
       <th>Type</th>
     </tr>
   </thead>
   <tr>
       <td><code>random_sampling</code></td>
       <td>Random samples</td>
       <td>Numpy array</td>
   </tr>
</table>

<h3>Model parameters</h3>
<br>
<p align = "justify" id = "mode">
    See examples of model parameters.
</p>

<h4><i>Crude Monte Carlo</i></h4>

```python
model = {'model sampling': 'mcs'}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'model sampling'</code></td>
        <td>Numerical algorithm used in sampling generator</td>
        <td>String</td>
    </tr>
</table>

<h4><i>Crude Monte Carlo - Stochastic analysis</i></h4>

```python
model = {'model sampling': 'mcs-time', 'time steps': 5}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'model sampling'</code></td>
        <td>Numerical algorithm used in sampling generator</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'time steps'</code></td>
        <td>Number of time steps used in times series analysis</td>
        <td>Integer</td>
    </tr>
</table>

<h3>Random variable parameters</h3>
<br>
<p align = "justify" id = "vars">
    See examples of variables parameters.
</p>

<h4><i>Normal sampling</i></h4>

```python
var = {'type': 'normal', 'loc': 40.3, 'scale': 4.64, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'seed'</code></td>
        <td>Random seed. Use <code>None</code> for random seed</td>
        <td>Integer</td>
    </tr>
</table>

<h4><i>Normal sampling in time series</i></h4>

```python
var = {'type': 'normal', 'loc': 40.3, 'scale': 4.64, 'stochastic variable': False, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
      <td><code>'seed'</code></td>
      <td>Random seed. Use <code>None</code> for random seed</td>
      <td>Integer</td>
    </tr>
    <tr>
        <td><code>'stochastic variable'</code></td>
        <td><p align = "justify">This variable represents the behavior of the random variable over time intervals. <code>False</code> means the variable with a fixed value in each time interval, and <code>True</code> represents the random behavior of the variable in each time interval.</p></td>
        <td>Boolean</td>
    </tr>
</table>

<h4><i>Triangular sampling</i></h4>

```python
var = {'type': 'triangular', 'min': 3, 'loc': 7, 'max': 8, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'min'</code></td>
        <td>Minimum value</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'max'</code></td>
        <td>Maximum value</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'seed'</code></td>
        <td>Random seed. Use <code>None</code> for random seed</td>
        <td>Integer</td>
    </tr>
</table>

<h4><i>Triangular sampling in time series</i></h4>

```python
var = {'type': 'triangular', 'min': 3, 'loc': 7, 'max': 8, 'stochastic variable': False, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'min'</code></td>
        <td>Minimum value</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'max'</code></td>
        <td>Maximum value</td>
        <td>Float</td>
    </tr>
    <tr>
      <td><code>'seed'</code></td>
      <td>Random seed. Use <code>None</code> for random seed</td>
      <td>Integer</td>
    </tr>
    <tr>
        <td><code>'stochastic variable'</code></td>
        <td><p align = "justify">This variable represents the behavior of the random variable over time intervals. <code>False</code> means the variable with a fixed value in each time interval, and <code>True</code> represents the random behavior of the variable in each time interval.</p></td>
        <td>Boolean</td>
    </tr>
</table>

<h4><i>Gumbel max. sampling</i></h4>

```python
var = {'type': 'gumbel max', 'loc': 40.3, 'scale': 4.64, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'seed'</code></td>
        <td>Random seed. Use <code>None</code> for random seed</td>
        <td>Integer</td>
    </tr>
</table>

<h4><i>Gumbel max. sampling in time series</i></h4>

```python
var = {'type': 'gumbel max', 'loc': 40.3, 'scale': 4.64, 'stochastic variable': False, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
      <td><code>'seed'</code></td>
      <td>Random seed. Use <code>None</code> for random seed</td>
      <td>Integer</td>
    </tr>
    <tr>
        <td><code>'stochastic variable'</code></td>
        <td><p align = "justify">This variable represents the behavior of the random variable over time intervals. <code>False</code> means the variable with a fixed value in each time interval, and <code>True</code> represents the random behavior of the variable in each time interval.</p></td>
        <td>Boolean</td>
    </tr>
</table>

<h4><i>Gumbel min. sampling</i></h4>

```python
var = {'type': 'gumbel min', 'loc': 40.3, 'scale': 4.64, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'seed'</code></td>
        <td>Random seed. Use <code>None</code> for random seed</td>
        <td>Integer</td>
    </tr>
</table>

<h4><i>Gumbel min. sampling in time series</i></h4>

```python
var = {'type': 'gumbel min', 'loc': 40.3, 'scale': 4.64, 'stochastic variable': False, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
      <td><code>'seed'</code></td>
      <td>Random seed. Use <code>None</code> for random seed</td>
      <td>Integer</td>
    </tr>
    <tr>
        <td><code>'stochastic variable'</code></td>
        <td><p align = "justify">This variable represents the behavior of the random variable over time intervals. <code>False</code> means the variable with a fixed value in each time interval, and <code>True</code> represents the random behavior of the variable in each time interval.</p></td>
        <td>Boolean</td>
    </tr>
</table>

<h4><i>Log-normal sampling</i></h4>

```python
var = {'type': 'lognormal', 'loc': 40.3, 'scale': 4.64, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'seed'</code></td>
        <td>Random seed. Use <code>None</code> for random seed</td>
        <td>Integer</td>
    </tr>
</table>

<h4><i>Log-normal sampling in time series</i></h4>

```python
var = {'type': 'lognormal', 'loc': 40.3, 'scale': 4.64, 'stochastic variable': False, 'seed': None}
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead> 
    <tr>
        <td><code>'type'</code></td>
        <td>Type of probability density function</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'loc'</code></td>
        <td>Mean</td>
        <td>Float</td>
    </tr>
    <tr>
        <td><code>'scale'</code></td>
        <td>Standard deviation</td>
        <td>Float</td>
    </tr>
    <tr>
      <td><code>'seed'</code></td>
      <td>Random seed. Use <code>None</code> for random seed</td>
      <td>Integer</td>
    </tr>
    <tr>
        <td><code>'stochastic variable'</code></td>
        <td><p align = "justify">This variable represents the behavior of the random variable over time intervals. <code>False</code> means the variable with a fixed value in each time interval, and <code>True</code> represents the random behavior of the variable in each time interval.</p></td>
        <td>Boolean</td>
    </tr>
</table>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, we will use the <code>sampling</code> function to generate a set of random samples \((n=1000000)\) following a Normal distribution with mean \(\mu = 7\) and standard deviation \(\sigma = 3\). Use "seed without control" in your setup.</i>
</p>

```python
from parepy_toolbox import sampling

# Dataset
model = {'model sampling': 'mcs'}
f = {'type': 'normal', 'loc': 7, 'scale': 3, 'seed': None}
varSet = [f]
size = 1000000

# Call function
r = sampling(size, len(varSet), model, varSet)

# Output details
print(f'random variables n={len(r)}: \n\n {r} \n\n type variable: {type(r)}')
```

```bash
random variables n=1000000: 

 [[ 3.5106688 ]
 [ 5.69074161]
 [ 8.60932312]
 ...
 [10.28841779]
 [ 7.25655855]
 [ 7.21348877]] 

 type variable: <class 'numpy.ndarray'>
```

Example 2
{: .label .label-blue }

<p align = "justify">
    <i>In this example, we will use the <code>sampling</code> function to generate a set of random samples \((n=30)\) following a Normal distribution with mean \(\mu = 7\) and standard deviation \(\sigma = 3\). Use "stochastic variable" in your setup.</i>
</p>

```python
from parepy_toolbox import sampling

# Dataset: fixed value in time series
model = {'model sampling': 'mcs-time', 'time steps': 3}
f = {'type': 'normal', 'loc': 7, 'scale': 3, 'seed': None, 'stochastic variable': False}
varSet = [f]
size = 30

# Call function
r = sampling(size, len(varSet), model, varSet)
```

```bash
[[ 9.83084403  0.        ]  --> sample 0 time step = 0
 [ 9.83084403  1.        ]  --> sample 0 time step = 1
 [ 9.83084403  2.        ]  --> sample 0 time step = 2
 [10.21614749  0.        ]  --> sample 1 time step = 0
 [10.21614749  1.        ]  --> sample 1 time step = 1
 [10.21614749  2.        ]  --> sample 1 time step = 2
 [10.06481753  0.        ]  --> sample 2 time step = 0
 [10.06481753  1.        ]  --> sample 2 time step = 1
 [10.06481753  2.        ]  --> sample 2 time step = 2
```

Example 3
{: .label .label-blue }

<p align = "justify">
    <i>In this example, we will use the <code>sampling</code> function to generate a set of random samples \((n=30)\) following a Normal distribution with mean \(\mu = 7\) and standard deviation \(\sigma = 3\). Use "stochastic variable" in your setup.</i>
</p>

```python
from parepy_toolbox import sampling

# Dataset stochastic value in time series
model = {'model sampling': 'mcs-time', 'time steps': 3}
f = {'type': 'normal', 'loc': 7, 'scale': 3, 'seed': None, 'stochastic variable': True}
varSet = [f]
size = 30

# Call function
r = sampling(size, len(varSet), model, varSet)
print(r)
```

```bash
[[7.73489316  0.        ]  --> sample 0 time step = 0
 [2.39698123  1.        ]  --> sample 0 time step = 1
 [6.73925933  2.        ]  --> sample 0 time step = 2
 [5.09639542  0.        ]  --> sample 1 time step = 0
 [5.46050213  1.        ]  --> sample 1 time step = 1
 [6.77307753  2.        ]  --> sample 1 time step = 2
 [5.15205194  0.        ]  --> sample 2 time step = 0
 [6.95330405  1.        ]  --> sample 2 time step = 1
 [6.65245261  2.        ]  --> sample 2 time step = 2
```
<!-- <p align = "center"><b>Figure 3.</b> Gumbel Distribution using Sampling function.</p>
<center><img src="assets/images/lognormal_distribution.png" width="70%"></center> -->
