from parepy_toolbox import sampling_algorithm_structural_analysis, convergence_probability_failure, sobol_algorithm
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import textwrap
import json


def generate_function(capacity_expr, demand_expr):
    function_code = f"""
    def obj_function(x, none_variable):
        # Random variables
        f_y = x[0]
        p_load = x[1]
        w_load = x[2]
        
        capacity = {capacity_expr}
        demand = {demand_expr}

        # State limit function
        constraint = capacity - demand

        return [capacity], [demand], [constraint]
    """
    
    with open("obj_functions.py", "w") as f:
        f.write(textwrap.dedent(function_code))
    
    return function_code

st.title("PAREpy")
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed laoreet nisl quis quam mattis molestie. Aliquam efficitur, risus et fringilla pellentesque, est sapien finibus sapien, vitae scelerisque nisl nunc vel justo. Nullam ut ornare diam. Ut convallis ex velit, eu condimentum ligula porttitor nec. Sed id magna ut elit fermentum convallis. Curabitur tincidunt tellus tortor, et ultrices massa faucibus sit amet. Suspendisse aliquam, massa et posuere dictum, ipsum purus egestas leo, a placerat metus felis non magna. Fusce ac sem aliquam, egestas velit vel, laoreet mi. Nulla lacinia tortor id interdum faucibus. Ut laoreet felis at purus congue, eget viverra metus blandit. Donec placerat finibus laoreet. Quisque luctus sodales felis, in sollicitudin sem tristique eu. Aenean aliquet nunc sem, vel scelerisque nisi ornare eu. Nulla orci turpis, molestie non ex at, fringilla elementum enim. Cras dictum, dui nec tincidunt scelerisque, neque augue ullamcorper leo, sit amet vulputate ex diam vitae nisl.")

st.subheader("Objective Function parameters")
# Entrada do usuário
capacity_input = st.text_area("Capacity:", "80 * x[0]")
demand_input = st.text_area("Demand:", "54 * x[1] + 5832 * x[2]")


# Configuração do setup
st.subheader("")
st.subheader("Setup Configuration")
num_samples = st.number_input("Número de amostras", min_value=1, step=1, value=10000)
model_sampling = st.selectbox("Método de amostragem", ["mcs"], index=0)


st.subheader("")
st.subheader("Model Configuration")

# Lista para armazenar as variáveis
if "var" not in st.session_state:
    st.session_state.var = []

# Definir número de variáveis
num_vars = st.number_input("Número de variáveis aleatórias", min_value=1, step=1, value=max(1, len(st.session_state.var)))

# Ajustar o número de variáveis armazenadas
while len(st.session_state.var) < num_vars:
    st.session_state.var.append({
        'type': 'normal',
        'parameters': {'mean': 40.3, 'sigma': 4.64},
        'stochastic variable': False
    })
while len(st.session_state.var) > num_vars:
    st.session_state.var.pop()

# Opções de distribuição
distribution_types = ["uniform", "normal", "lognormal", "gumbel max", "gumbel min", "triangular"]

# Criar inputs para cada variável
with st.container():
    for i in range(num_vars):
        with st.expander(f"Variável {i+1}"):
            var_type = st.selectbox(f"Tipo da variável {i+1}", distribution_types, key=f"type_{i}", index=distribution_types.index(st.session_state.var[i]['type']))
            
            if var_type == "triangular":
                min_val = st.number_input(f"Mínimo da variável {i+1}", key=f"min_{i}", value=st.session_state.var[i]['parameters'].get('min', 0.0))
                mode = st.number_input(f"Moda da variável {i+1}", key=f"mode_{i}", value=st.session_state.var[i]['parameters'].get('mode', 0.0))
                max_val = st.number_input(f"Máximo da variável {i+1}", key=f"max_{i}", value=st.session_state.var[i]['parameters'].get('max', 0.0))
                parameters = {'min': min_val, 'mode': mode, 'max': max_val}
            else:
                mean = st.number_input(f"Média da variável {i+1}", key=f"mean_{i}", value=st.session_state.var[i]['parameters'].get('mean', 0.0))
                sigma = st.number_input(f"Sigma da variável {i+1}", key=f"sigma_{i}", value=st.session_state.var[i]['parameters'].get('sigma', 1.0))
                parameters = {'mean': mean, 'sigma': sigma}

            
            # Atualizar valores
            st.session_state.var[i] = {
                'type': var_type,
                'parameters': parameters,
            }

if st.button("Run Simulation"):
    function_str = generate_function(capacity_input, demand_input)

    from obj_functions import obj_function
    
    setup = {
        'number of samples': num_samples,
        'numerical model': {'model sampling': model_sampling},
        'variables settings': st.session_state.var,
        'number of state limit functions or constraints': 1,
        'none variable': None,
        'objective function': obj_function,
        'name simulation': None,
    }

    results, pf, beta = sampling_algorithm_structural_analysis(setup)

    # Gráficos
    st.session_state.text_convergence = "Convergence Rate:"
    div, m, ci_l, ci_u, var = convergence_probability_failure(results, 'G_0')
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(div, m, label="Failure Probability Rate", color='b', linestyle='-')
    ax1.fill_between(div, ci_l, ci_u, color='b', alpha=0.2, label="95% Confidence Interval")
    ax1.set_xlabel("Sample Size (div)")
    ax1.set_ylabel("Failure Probability Rate")
    ax1.set_title("Convergence of Failure Probability")
    ax1.legend()
    ax1.grid(True)
    st.session_state.fig1 = fig1

    st.session_state.text_sobol = "Sobol Sensitivity Analysis:"
    data_sobol = sobol_algorithm(setup)
    variables = ['x_0', 'x_1', 'x_2']
    s_i = [data_sobol.iloc[var]['s_i'] for var in range(len(variables))]
    s_t = [data_sobol.iloc[var]['s_t'] for var in range(len(variables))]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    x = range(len(variables))
    width = 0.35
    ax2.bar(x, s_i, width, label='First-order (s_i)', color='blue', alpha=0.7)
    ax2.bar([p + width for p in x], s_t, width, label='Total-order (s_t)', color='orange', alpha=0.7)
    ax2.set_xlabel("Variables")
    ax2.set_ylabel("Sobol Index")
    ax2.set_xticks([p + width / 2 for p in x])
    ax2.set_xticklabels(variables)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.session_state.fig2 = fig2

    st.session_state.results = results
    st.session_state.pf = pf
    st.session_state.beta = beta
    st.session_state.data_sobol = data_sobol

# Re-rendering everything, checking session state to ensure content persists
if "text_convergence" in st.session_state:
    st.subheader(st.session_state.text_convergence)

if "fig1" in st.session_state:
    st.pyplot(st.session_state.fig1)

if "text_sobol" in st.session_state:
    st.subheader(st.session_state.text_sobol)

if "fig2" in st.session_state:
    st.pyplot(st.session_state.fig2)

# Download
if "results" in st.session_state:
    results = st.session_state.results  # Access results from session state
    final_results = BytesIO()
    with pd.ExcelWriter(final_results, engine="xlsxwriter") as writer:
        results.to_excel(writer, index=False, sheet_name="Results")
    final_results.seek(0)
    st.download_button("Download Resultados", final_results, file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.error("Erro: os resultados não estão disponíveis para download. Verifique se o algoritmo foi executado corretamente.")





