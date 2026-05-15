import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import r2_score
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RBFInterpolator
import statsmodels.api as sm

st.set_page_config(layout="wide", page_title="RSM Optimization Toolkit")

# ---------- Helper function for plotting ----------
def plot_rsm_surface(predict_fn, x1_range, x2_range,
                     x1_label='X1', x2_label='X2', y_label='Response',
                     title='Response Surface', optimum=None, grid_n=60):
    x1 = np.linspace(*x1_range, grid_n)
    x2 = np.linspace(*x2_range, grid_n)
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1g.ravel(), X2g.ravel()])
    Yg = predict_fn(X_grid).reshape(X1g.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Contour
    ax = axes[0]
    cp = ax.contourf(X1g, X2g, Yg, levels=25, cmap='RdYlBu_r')
    ax.contour(X1g, X2g, Yg, levels=15, colors='k', linewidths=0.4)
    fig.colorbar(cp, ax=ax, shrink=0.85)
    if optimum:
        ax.plot(optimum['x1'], optimum['x2'], 'r*', ms=16, zorder=5,
                label=f"Opt ({optimum['x1']:.3f}, {optimum['x2']:.3f})")
        ax.legend(fontsize=9)
    ax.set_xlabel(x1_label); ax.set_ylabel(x2_label)
    ax.set_title(f'{title} — Contour')

    # 3D Surface
    ax3 = fig.add_subplot(122, projection='3d')
    axes[1].remove()
    ax3.plot_surface(X1g, X2g, Yg, cmap='RdYlBu_r', alpha=0.85,
                     edgecolor='k', linewidth=0.15)
    if optimum:
        ax3.scatter(optimum['x1'], optimum['x2'], optimum['y'],
                    color='red', s=120, marker='*', zorder=5)
    ax3.set_xlabel(x1_label); ax3.set_ylabel(x2_label); ax3.set_zlabel(y_label)
    ax3.set_title(f'{title} — Surface')
    ax3.view_init(elev=30, azim=225)
    plt.tight_layout()
    return fig

# ---------- Desirability functions ----------
def individual_desirability(y, spec):
    L, U, T, s = spec['low'], spec['high'], spec['target'], spec['shape']
    goal = spec['goal']
    if goal == 'maximize':
        if y < L:
            return 0.0
        elif y > T:
            return 1.0
        else:
            return ((y - L) / (T - L)) ** s
    elif goal == 'minimize':
        if y > U:
            return 0.0
        elif y < T:
            return 1.0
        else:
            return ((U - y) / (U - T)) ** s
    else:  # target
        if y < L or y > U:
            return 0.0
        elif y <= T:
            return ((y - L) / (T - L)) ** s
        else:
            return ((U - y) / (U - T)) ** s

def overall_desirability(x, models, specs):
    x_arr = np.array(x).reshape(1, -1)
    x_poly = poly_3.transform(x_arr)
    D = 1.0
    total_w = sum(s['weight'] for s in specs)
    for model, spec in zip(models, specs):
        y_pred = model.predict(x_poly)[0]
        d = individual_desirability(y_pred, spec)
        D *= d ** (spec['weight'] / total_w)
    return D

# ---------- Sidebar ----------
st.sidebar.title("RSM Optimization Toolkit")
method = st.sidebar.selectbox("Select Method",
                              ["1st-Order RSM", "2nd-Order RSM",
                               "Desirability Function", "Non-Linear RSM"])

# ---------- 1st-Order RSM ----------
if method == "1st-Order RSM":
    st.header("1st-Order RSM (Linear Model + Steepest Ascent)")
    st.markdown("Use when far from optimum for screening or steepest ascent path.")
    col1, col2 = st.columns(2)
    with col1:
        factor_names = st.text_input("Factor names (comma separated)", "Temperature (°C), Pressure (atm)").split(",")
        factor_names = [f.strip() for f in factor_names]
        natural_low = st.text_input("Natural low values (comma separated)", "150, 2")
        natural_low = list(map(float, natural_low.split(",")))
        natural_high = st.text_input("Natural high values (comma separated)", "190, 6")
        natural_high = list(map(float, natural_high.split(",")))
    with col2:
        goal = st.selectbox("Goal", ["maximize", "minimize"])
        n_steps = st.number_input("Number of steepest ascent steps", min_value=1, max_value=20, value=6)

    st.subheader("Coded Design Matrix and Responses")
    st.markdown("Enter each row as: x1, x2, y (comma separated)")
    default_data = """-1, -1, 52
1, -1, 74
-1, 1, 62
1, 1, 80
0, 0, 70
0, 0, 68
0, 0, 71"""
    data_str = st.text_area("Data (one row per experiment)", default_data, height=200)
    rows = [list(map(float, line.split(","))) for line in data_str.strip().split("\n")]
    X_coded = np.array([r[:2] for r in rows])
    y = np.array([r[2] for r in rows])

    if st.button("Run 1st-Order Analysis"):
        # Fit model
        X_const = sm.add_constant(X_coded)
        model = sm.OLS(y, X_const).fit()
        st.subheader("Regression Results")
        st.text(model.summary2())
        st.write(f"R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")

        # Steepest ascent/descent
        betas = model.params[1:]
        direction = betas / np.linalg.norm(betas)
        if goal == 'minimize':
            direction = -direction
        step_size = 1.0 / np.max(np.abs(direction)) * direction
        center_natural = (np.array(natural_low) + np.array(natural_high)) / 2
        half_range = (np.array(natural_high) - np.array(natural_low)) / 2

        steps = []
        for k in range(n_steps + 1):
            coded_pt = k * step_size
            natural_pt = center_natural + coded_pt * half_range
            steps.append([k, *coded_pt, *natural_pt])
        path_df = pd.DataFrame(steps,
                               columns=['Step', f'{factor_names[0]} (coded)', f'{factor_names[1]} (coded)',
                                        f'{factor_names[0]} (natural)', f'{factor_names[1]} (natural)'])
        st.subheader("Steepest Ascent/Descent Path")
        st.dataframe(path_df)

        # Surface plot
        def predict_fn(X):
            return model.predict(sm.add_constant(X, has_constant='add'))
        fig = plot_rsm_surface(predict_fn, x1_range=(-2,2), x2_range=(-2,2),
                               x1_label=factor_names[0]+' (coded)',
                               x2_label=factor_names[1]+' (coded)',
                               title='1st-Order RSM')
        st.pyplot(fig)

# ---------- 2nd-Order RSM ----------
elif method == "2nd-Order RSM":
    st.header("2nd-Order RSM (Quadratic Model)")
    st.markdown("Used near optimum to characterize curvature. Requires CCD design.")
    col1, col2 = st.columns(2)
    with col1:
        factor_names = st.text_input("Factor names (comma separated)", "Cutting Speed, Feed Rate").split(",")
        factor_names = [f.strip() for f in factor_names]
        natural_low = st.text_input("Natural low values", "100, 0.05")
        natural_low = list(map(float, natural_low.split(",")))
        natural_high = st.text_input("Natural high values", "200, 0.25")
        natural_high = list(map(float, natural_high.split(",")))
        goal = st.selectbox("Goal", ["maximize", "minimize"])
    with col2:
        alpha = np.sqrt(2)
        st.write(f"CCD alpha = √2 = {alpha:.3f} (rotatable)")
        st.info("The CCD design matrix is automatically generated for 2 factors.")
        # Default CCD for 2 factors
        X_coded = np.array([
            [-1, -1], [1, -1], [-1, 1], [1, 1],
            [-alpha, 0], [alpha, 0], [0, -alpha], [0, alpha],
            [0,0], [0,0], [0,0]
        ])
        st.write("Coded design matrix (11 runs):")
        st.dataframe(pd.DataFrame(X_coded, columns=[f"{fn} (coded)" for fn in factor_names]))
    st.subheader("Observed Responses (in same order as design)")
    default_y = "3.8,2.5,5.2,3.1,4.9,2.0,3.2,4.8,1.5,1.6,1.4"
    y_str = st.text_area("Response values (comma separated)", default_y)
    y = np.array(list(map(float, y_str.split(","))))

    if st.button("Run 2nd-Order Analysis"):
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X_coded)
        model = sm.OLS(y, X_poly).fit()
        st.subheader("Quadratic Model Coefficients")
        coeff_df = pd.DataFrame({
            'Term': poly.get_feature_names_out(['x1','x2']),
            'Coefficient': model.params,
            'Std Error': model.bse,
            'p-value': model.pvalues
        })
        st.dataframe(coeff_df)
        st.write(f"R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")

        # Stationary point analysis
        b = model.params
        b_vec = b[1:3]
        B = np.array([[b[3], b[4]/2], [b[4]/2, b[5]]])
        try:
            x_stat = -0.5 * np.linalg.solve(B, b_vec)
            y_stat = model.predict(poly.transform(x_stat.reshape(1,-1)))[0]
            eigvals = np.linalg.eigvalsh(B)
            if all(eigvals > 0):
                nature = 'MINIMUM'
            elif all(eigvals < 0):
                nature = 'MAXIMUM'
            else:
                nature = 'SADDLE POINT'
            center_2 = (np.array(natural_low) + np.array(natural_high))/2
            half_2 = (np.array(natural_high) - np.array(natural_low))/2
            x_stat_natural = center_2 + x_stat * half_2
            st.subheader("Stationary Point")
            st.write(f"Coded: x1={x_stat[0]:.4f}, x2={x_stat[1]:.4f}")
            st.write(f"Natural: {factor_names[0]}={x_stat_natural[0]:.3f}, {factor_names[1]}={x_stat_natural[1]:.4f}")
            st.write(f"Predicted response: ŷ = {y_stat:.4f}")
            st.write(f"Eigenvalues: {eigvals}, Nature: {nature}")
        except:
            st.warning("B matrix singular; no unique stationary point.")

        # Bounded numerical optimization
        coded_bounds = [(-alpha, alpha), (-alpha, alpha)]
        def quad_predict(x):
            return model.predict(poly.transform(np.array(x).reshape(1,-1)))[0]
        sign = 1 if goal == 'minimize' else -1
        result = differential_evolution(lambda x: sign * quad_predict(x), bounds=coded_bounds, seed=42)
        opt_x2 = result.x
        opt_y2 = quad_predict(opt_x2)
        opt_natural = center_2 + opt_x2 * half_2
        st.subheader(f"Bounded Optimum ({goal})")
        st.write(f"Coded: x1={opt_x2[0]:.4f}, x2={opt_x2[1]:.4f}")
        st.write(f"Natural: {factor_names[0]}={opt_natural[0]:.3f}, {factor_names[1]}={opt_natural[1]:.4f}")
        st.write(f"Predicted response: ŷ = {opt_y2:.4f}")

        # Surface plot
        def predict_fn(X):
            return model.predict(poly.transform(X))
        fig = plot_rsm_surface(predict_fn, x1_range=(-alpha-0.3, alpha+0.3),
                               x2_range=(-alpha-0.3, alpha+0.3),
                               x1_label=factor_names[0]+' (coded)',
                               x2_label=factor_names[1]+' (coded)',
                               title='2nd-Order RSM',
                               optimum={'x1': opt_x2[0], 'x2': opt_x2[1], 'y': opt_y2})
        st.pyplot(fig)

        # Diagnostic plots
        fig, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].scatter(model.fittedvalues, model.resid, edgecolors='k', facecolors='steelblue')
        axes[0].axhline(0, color='red', ls='--')
        axes[0].set_xlabel('Fitted'); axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals vs Fitted')
        sm.qqplot(model.resid, line='45', ax=axes[1])
        axes[2].scatter(y, model.fittedvalues, edgecolors='k', facecolors='steelblue')
        lims = [min(y.min(), model.fittedvalues.min())-0.5, max(y.max(), model.fittedvalues.max())+0.5]
        axes[2].plot(lims, lims, 'r--')
        axes[2].set_xlabel('Actual'); axes[2].set_ylabel('Predicted')
        axes[2].set_title('Predicted vs Actual')
        plt.tight_layout()
        st.pyplot(fig)

# ---------- Desirability Function ----------
elif method == "Desirability Function":
    st.header("Desirability Function (Multi-Response Optimization)")
    st.markdown("Combine multiple responses into one overall desirability D.")
    col1, col2 = st.columns(2)
    with col1:
        factor_names = st.text_input("Factor names (comma separated)", "Temperature, Catalyst").split(",")
        factor_names = [f.strip() for f in factor_names]
        natural_low = st.text_input("Natural low values", "100, 0.05")
        natural_low = list(map(float, natural_low.split(",")))
        natural_high = st.text_input("Natural high values", "200, 0.25")
        natural_high = list(map(float, natural_high.split(",")))
    with col2:
        alpha = np.sqrt(2)
        st.info("CCD design generated automatically.")
        X_coded = np.array([
            [-1, -1], [1, -1], [-1, 1], [1, 1],
            [-alpha,0], [alpha,0], [0,-alpha], [0,alpha],
            [0,0], [0,0], [0,0]
        ])
    st.subheader("Response Specifications")
    st.markdown("Define each response: name, goal (maximize/minimize/target), low, high, target, weight, shape.")
    
    # Set initial number of responses to 3 to match default data
    if 'num_responses' not in st.session_state:
        st.session_state.num_responses = 3
    
    col_add = st.columns([1,1,2])
    if col_add[0].button("+ Add Response"):
        st.session_state.num_responses += 1
    if col_add[1].button("- Remove Last"):
        st.session_state.num_responses = max(1, st.session_state.num_responses - 1)
    if col_add[2].button("Reset to 3 Responses"):
        st.session_state.num_responses = 3

    specs = []
    for i in range(st.session_state.num_responses):
        with st.expander(f"Response {i+1}", expanded=(i<3)):
            default_name = ["Yield (%)", "Impurity (ppm)", "Viscosity (cP)"][i] if i < 3 else f"Response {i+1}"
            default_goal = ["maximize", "minimize", "target"][i] if i < 3 else "maximize"
            default_low = [65, 5, 30][i] if i < 3 else 0.0
            default_high = [95, 60, 70][i] if i < 3 else 100.0
            default_target = [95, 5, 50][i] if i < 3 else 50.0
            
            name = st.text_input(f"Name", default_name, key=f"name_{i}")
            goal = st.selectbox(f"Goal", ["maximize", "minimize", "target"], 
                               index=["maximize","minimize","target"].index(default_goal), key=f"goal_{i}")
            low = st.number_input(f"Low (L)", value=default_low, key=f"low_{i}")
            high = st.number_input(f"High (U)", value=default_high, key=f"high_{i}")
            target = st.number_input(f"Target (T)", value=default_target, key=f"target_{i}")
            weight = st.number_input(f"Weight", value=1.0, step=0.1, key=f"weight_{i}")
            shape = st.number_input(f"Shape (s)", value=1.0, step=0.1, key=f"shape_{i}")
            specs.append({
                'name': name, 'goal': goal, 'low': low, 'high': high,
                'target': target, 'weight': weight, 'shape': shape
            })

    st.subheader("Response Data")
    st.markdown("Enter responses for each run (same order as CCD design). One line per response, values comma-separated.")
    default_data = """72,83,77,89,68,87,75,82,91,90,92
45,30,50,22,55,18,40,35,12,14,11
35,55,30,62,28,65,40,48,52,50,51"""
    data_lines = st.text_area("Responses (one line per response, comma separated)", default_data, height=200)
    
    # Parse response data
    response_data = []
    for line in data_lines.strip().split("\n"):
        if line.strip():
            response_data.append(list(map(float, line.split(","))))
    
    # Check if number of lines matches number of responses
    if len(response_data) != st.session_state.num_responses:
        st.error(f"Number of response lines ({len(response_data)}) does not match number of responses ({st.session_state.num_responses}). Please adjust the number of responses or the data.")
    else:
        # Ensure all response lines have the same length (should be 11 for CCD)
        expected_len = len(X_coded)
        for i, resp in enumerate(response_data):
            if len(resp) != expected_len:
                st.error(f"Response {i+1} has {len(resp)} values, but expected {expected_len} (matching design matrix).")
                st.stop()
        
        all_responses = np.array(response_data).T  # each column is a response
        
# Inside Desirability Function, after fitting models, replace the optimization block with:

if st.button("Run Desirability Optimization"):
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_full = poly.fit_transform(X_coded)
    models = []
    for i in range(st.session_state.num_responses):
        ols = sm.OLS(all_responses[:, i], X_full).fit()
        models.append(ols)
        st.write(f"Model for {specs[i]['name']}: R² = {ols.rsquared:.4f}")
    
    # Bounds for optimization
    bounds_3 = [(-alpha, alpha), (-alpha, alpha)]
    
    # Robust desirability function with error handling
    def safe_overall_desirability(x, models, specs, poly):
        try:
            x_arr = np.array(x).reshape(1, -1)
            x_poly = poly.transform(x_arr)
            D = 1.0
            total_w = sum(s['weight'] for s in specs)
            for model, spec in zip(models, specs):
                y_pred = model.predict(x_poly)[0]
                d = individual_desirability(y_pred, spec)
                # Avoid zero or negative values that break geometric mean
                d = max(d, 1e-6)
                D *= d ** (spec['weight'] / total_w)
            return float(D)
        except Exception as e:
            return 0.0  # Return very low desirability for invalid points
    
    # Optimize
    result = differential_evolution(
        lambda x: -safe_overall_desirability(x, models, specs, poly),
        bounds=bounds_3,
        seed=42,
        maxiter=500,
        popsize=15,
        disp=False
    )
    
    if result.success:
        opt_x = result.x
        opt_D = -result.fun
    else:
        st.error("Optimization failed to converge. Try different initial bounds or increase maxiter.")
        opt_x = result.x  # Still use the best found
        opt_D = -result.fun
    
    # Compute natural optimum
    center_2 = (np.array(natural_low) + np.array(natural_high))/2
    half_2 = (np.array(natural_high) - np.array(natural_low))/2
    opt_nat = center_2 + opt_x * half_2
    
    st.subheader("Optimal Solution")
    st.write(f"Coded optimum: x1={opt_x[0]:.4f}, x2={opt_x[1]:.4f}")
    st.write(f"Natural optimum: {factor_names[0]}={opt_nat[0]:.3f}, {factor_names[1]}={opt_nat[1]:.4f}")
    st.write(f"Overall Desirability D = {opt_D:.4f}")
    
    # Individual responses at optimum
    x_opt_poly = poly.transform(opt_x.reshape(1,-1))
    st.write("Individual responses at optimum:")
    for i, model in enumerate(models):
        y_pred = model.predict(x_opt_poly)[0]
        d_i = individual_desirability(y_pred, specs[i])
        st.write(f"{specs[i]['name']}: ŷ = {y_pred:.3f}  →  d = {d_i:.4f}")
    
    # Desirability surface plot
    def predict_desirability(X):
        return np.array([safe_overall_desirability(x, models, specs, poly) for x in X])
    
    fig = plot_rsm_surface(predict_desirability,
                           x1_range=(-alpha,alpha), x2_range=(-alpha,alpha),
                           x1_label=factor_names[0]+' (coded)',
                           x2_label=factor_names[1]+' (coded)',
                           y_label='Desirability D',
                           title='Overall Desirability',
                           optimum={'x1': opt_x[0], 'x2': opt_x[1], 'y': opt_D})
    st.pyplot(fig)
    
    # Bar chart
    fig2, ax = plt.subplots(figsize=(8,4))
    names = [s['name'] for s in specs] + ['Overall D']
    d_vals = [individual_desirability(models[i].predict(x_opt_poly)[0], specs[i]) for i in range(len(specs))] + [opt_D]
    colors = ['#4C72B0'] * len(specs) + ['#8172B2']
    bars = ax.barh(names, d_vals, color=colors, edgecolor='k', height=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Desirability')
    ax.set_title('Individual & Overall Desirabilities')
    for bar, val in zip(bars, d_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    plt.tight_layout()
    st.pyplot(fig2)
    
# ---------- Non-Linear RSM ----------
elif method == "Non-Linear RSM":
    st.header("Non-Linear RSM (Gaussian Process / RBF)")
    st.markdown("Use when polynomial models are inadequate. Supports Gaussian Process (Kriging) or RBF interpolation.")
    col1, col2 = st.columns(2)
    with col1:
        factor_names = st.text_input("Factor names (comma separated)", "pH, Substrate (mM)").split(",")
        factor_names = [f.strip() for f in factor_names]
        bounds_low = st.text_input("Bounds low values", "4.0, 5.0")
        bounds_low = list(map(float, bounds_low.split(",")))
        bounds_high = st.text_input("Bounds high values", "8.0, 25.0")
        bounds_high = list(map(float, bounds_high.split(",")))
        goal = st.selectbox("Goal", ["maximize", "minimize"])
        model_type = st.selectbox("Surrogate Model", ["Gaussian Process", "RBF Interpolator"])
    with col2:
        st.subheader("Data Points (natural units)")
        default_data = """4.0,5,12
5.0,5,35
6.0,5,68
7.0,5,45
8.0,5,15
4.0,15,18
5.0,15,52
6.0,15,85
7.0,15,60
8.0,15,20
4.0,25,14
5.0,25,40
6.0,25,72
7.0,25,50
8.0,25,16
6.0,10,78
6.0,20,80
5.5,12,65
6.5,18,75"""
        data_str = st.text_area("Data (x1, x2, y) per line", default_data, height=300)
        rows = [list(map(float, line.split(","))) for line in data_str.strip().split("\n")]
        X_nat = np.array([r[:2] for r in rows])
        y = np.array([r[2] for r in rows])
        st.write(f"Loaded {len(y)} data points.")

    if st.button("Run Non-Linear Analysis"):
        bounds = list(zip(bounds_low, bounds_high))
        sign = 1 if goal == 'minimize' else -1

        if model_type == "Gaussian Process":
            kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 5.0])
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15,
                                           alpha=1e-6, normalize_y=True)
            gpr.fit(X_nat, y)
            y_pred = gpr.predict(X_nat)
            r2 = r2_score(y, y_pred)
            st.write(f"Kernel: {gpr.kernel_}")
            st.write(f"Log-marginal-likelihood: {gpr.log_marginal_likelihood_value_:.3f}")
            st.write(f"R² (training): {r2:.4f}")
            # Optimize
            res = differential_evolution(
                lambda x: sign * gpr.predict(x.reshape(1,-1))[0],
                bounds=bounds, seed=42, maxiter=1000
            )
            opt_x = res.x
            opt_y, opt_std = gpr.predict(opt_x.reshape(1,-1), return_std=True)
            st.subheader("Optimization Results")
            st.write(f"Optimum: {factor_names[0]} = {opt_x[0]:.3f}, {factor_names[1]} = {opt_x[1]:.3f}")
            st.write(f"Predicted response: ŷ = {opt_y[0]:.3f}  ± {opt_std[0]:.3f} (1σ)")
            # Surface plots using GPR
            def predict_gpr(X):
                return gpr.predict(X)
            fig = plot_rsm_surface(predict_gpr,
                                   x1_range=bounds[0], x2_range=bounds[1],
                                   x1_label=factor_names[0], x2_label=factor_names[1],
                                   y_label='Response', title='Gaussian Process Surface',
                                   optimum={'x1': opt_x[0], 'x2': opt_x[1], 'y': opt_y[0]})
            st.pyplot(fig)
        else:  # RBF Interpolator
            rbf = RBFInterpolator(X_nat, y, kernel='thin_plate_spline', smoothing=1.0)
            y_pred = rbf(X_nat)
            r2 = r2_score(y, y_pred)
            st.write(f"RBF Interpolator (thin-plate spline)")
            st.write(f"R² (training): {r2:.4f}")
            # Optimize
            res = differential_evolution(
                lambda x: sign * rbf(x.reshape(1,-1))[0],
                bounds=bounds, seed=42, maxiter=1000
            )
            opt_x = res.x
            opt_y = rbf(opt_x.reshape(1,-1))[0]
            st.subheader("Optimization Results")
            st.write(f"Optimum: {factor_names[0]} = {opt_x[0]:.3f}, {factor_names[1]} = {opt_x[1]:.3f}")
            st.write(f"Predicted response: ŷ = {opt_y:.3f}")
            def predict_rbf(X):
                return rbf(X)
            fig = plot_rsm_surface(predict_rbf,
                                   x1_range=bounds[0], x2_range=bounds[1],
                                   x1_label=factor_names[0], x2_label=factor_names[1],
                                   y_label='Response', title='RBF Interpolation Surface',
                                   optimum={'x1': opt_x[0], 'x2': opt_x[1], 'y': opt_y})
            st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Upload data or enter manually. For more details, refer to the original notebook.")