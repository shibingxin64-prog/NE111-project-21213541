import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="NE111 Project", layout="wide")

st.title("NE111 Project")
st.write(
    "This app fits histograms of your data to distributions from `scipy.stats` "
    "and provides both automatic and manual fitting options."
)

def parse_text_data(text):
    text = text.strip()
    if text == "":
        return np.array([])
    for sep in [",", ";", "\t"]:
        text = text.replace(sep, " ")
    parts = text.split()
    values = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            pass
    return np.array(values)

def compute_errors(data, dist, bins):
    if data.size == 0:
        return np.nan, np.nan
    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf_vals = dist.pdf(centers)
    diff = hist - pdf_vals
    mse = float(np.mean(diff ** 2))
    max_err = float(np.max(np.abs(diff)))
    return mse, max_err

DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma (gamma)": stats.gamma,
    "Exponential (expon)": stats.expon,
    "Weibull (weibull_min)": stats.weibull_min,
    "Lognormal (lognorm)": stats.lognorm,
    "Beta (beta)": stats.beta,
    "Uniform (uniform)": stats.uniform,
    "Chi-squared (chi2)": stats.chi2,
    "Rayleigh (rayleigh)": stats.rayleigh,
    "Logistic (logistic)": stats.logistic,
}

st.sidebar.header("Data Input")
input_mode = st.sidebar.radio(
    "How do you want to provide data?",
    ["Paste data", "Upload CSV"]
)

data = np.array([])

if input_mode == "Paste data":
    default_text = "1.2 1.4 1.5 1.7 2.0 2.1 2.4 2.5 3.0"
    text_data = st.sidebar.text_area(
        "Paste numeric values:",
        value=default_text,
        height=150
    )
    data = parse_text_data(text_data)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Detected columns:", list(df.columns))
            column = st.sidebar.selectbox("Choose a numeric column:", df.columns)
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            data = series.to_numpy()
        except Exception as e:
            st.sidebar.error("Error reading CSV file: {}".format(e))
    else:
        st.sidebar.info("Please upload a CSV file.")

st.sidebar.header("Distribution & Options")
dist_name = st.sidebar.selectbox("Choose a distribution:", list(DISTRIBUTIONS.keys()))
dist_obj = DISTRIBUTIONS[dist_name]
manual_mode = st.sidebar.checkbox("Enable manual fitting sliders", value=False)
n_bins = st.sidebar.slider("Number of histogram bins:", 5, 100, 25)

if data.size == 0:
    st.warning("No data yet. Please paste numbers or upload a CSV file.")
    st.stop()

st.success("Loaded {} data points.".format(data.size))

with st.expander("Basic statistics"):
    st.write("Mean:", float(np.mean(data)))
    st.write("Std dev:", float(np.std(data)))
    st.write("Min:", float(np.min(data)))
    st.write("Max:", float(np.max(data)))

try:
    params = dist_obj.fit(data)
except Exception as e:
    st.error("Could not fit distribution {}: {}".format(dist_name, e))
    st.stop()

n_params = len(params)
n_shape = max(0, n_params - 2)
shape_params = list(params[:n_shape])
loc_param = float(params[-2])
scale_param = float(params[-1])

param_names = ["shape{}".format(i + 1) for i in range(n_shape)] + ["loc", "scale"]
param_values = [float(v) for v in params]
param_table = pd.DataFrame({"Parameter": param_names, "Value": param_values})

tab_plot, tab_details = st.tabs(["Visualization", "Fit details"])

with tab_details:
    st.subheader("Automatic fit parameters")
    st.table(param_table)

st.subheader("Manual fitting controls")
if manual_mode:
    manual_shapes = []
    for i in range(n_shape):
        p = float(shape_params[i])
        if p > 0:
            low = max(0.0001, p * 0.1)
        else:
            low = 0.0001
        high = max(low + 0.1, p * 3.0 + 1.0)
        manual_shapes.append(
            st.slider(
                "shape{}".format(i + 1),
                float(low),
                float(high),
                p,
                key="shape{}".format(i + 1)
            )
        )
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    loc_low = data_min - abs(data_max - data_min)
    loc_high = data_max + abs(data_max - data_min)
    manual_loc = st.slider("loc", float(loc_low), float(loc_high), loc_param)
    base_scale = abs(scale_param)
    if base_scale == 0:
        base_scale = max(1.0, float(np.std(data)))
    scale_low = base_scale * 0.1
    scale_high = base_scale * 3.0
    manual_scale = st.slider("scale", float(scale_low), float(scale_high), scale_param)
    manual_params = tuple(manual_shapes + [manual_loc, manual_scale])
else:
    manual_params = None

data_min = float(np.min(data))
data_max = float(np.max(data))
x_min = data_min - 0.1 * abs(data_min)
x_max = data_max + 0.1 * abs(data_max)
if x_min == x_max:
    x_min -= 1.0
    x_max += 1.0
x = np.linspace(x_min, x_max, 400)

auto_dist = dist_obj(*params)
auto_pdf = auto_dist.pdf(x)
auto_mse, auto_max_err = compute_errors(data, auto_dist, bins=n_bins)

if manual_params is not None:
    manual_dist = dist_obj(*manual_params)
    manual_pdf = manual_dist.pdf(x)
    manual_mse, manual_max_err = compute_errors(data, manual_dist, bins=n_bins)
else:
    manual_dist = None
    manual_pdf = None
    manual_mse, manual_max_err = None, None

with tab_plot:
    st.subheader("Histogram and fitted distribution")
    col_plot, col_info = st.columns([2, 1])
    with col_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            data,
            bins=n_bins,
            density=True,
            alpha=0.5,
            edgecolor="k",
            color="green",
            label="Data (green bars)"
        )
        ax.plot(x, auto_pdf, linewidth=2, label="Automatic fit")
        if manual_pdf is not None:
            ax.plot(x, manual_pdf, linestyle="--", linewidth=2, label="Manual fit")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(dist_name)
        ax.legend()
        st.pyplot(fig)
    with col_info:
        st.markdown("**Fit quality metrics**")
        st.write("Automatic fit MSE:", auto_mse)
        st.write("Automatic fit max error:", auto_max_err)
        if manual_dist is not None:
            st.write("Manual fit MSE:", manual_mse)
            st.write("Manual fit max error:", manual_max_err)
            st.caption("Smaller values mean a closer match between histogram and curve.")
        else:
            st.write("Manual fitting disabled")

with tab_details:
    st.subheader("Fit quality metrics")
    st.write("Automatic fit MSE:", auto_mse)
    st.write("Automatic fit max error:", auto_max_err)
    if manual_dist is not None:
        st.write("Manual fit MSE:", manual_mse)
        st.write("Manual fit max error:", manual_max_err)

