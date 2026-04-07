import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@st.cache_data 
def generate_data(n_samples=50,noise_level=10.0,add_outliers=False):
    np.random.seed(42)
    X = np.linspace(0,100,n_samples)
    y = 3 * X + 20 + np.random.normal(0, noise_level, n_samples)

    if add_outliers:
        outlier_indices = np.random.choice(n_samples, 3, replace=False)
        y[outlier_indices] = y[outlier_indices] + np.random.choice([-150, 150], 3)
        
    return X, y

st.sidebar.header("Control Panel")

st.sidebar.subheader("1. Dataset")
noise_level = st.sidebar.slider("Noise Level", 0.0, 50.0, 15.0)
add_outliers = st.sidebar.toggle("Inject Outliers")

X, y = generate_data(noise_level=noise_level, add_outliers=add_outliers)

st.sidebar.subheader("2. Model Parameters")
m = st.sidebar.slider("Slope (m)", -10.0, 10.0, 0.0, 0.1)
b = st.sidebar.slider("Intercept (b)", -100.0, 100.0, 0.0, 1.0)

st.sidebar.subheader("3. Optimization")
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1])
iterations = st.sidebar.slider("Gradient Descent Steps", 1, 100, 10)
run_gd = st.sidebar.button("Run Gradient Descent")

st.title("Interactive Linear Regression")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Data & Line", "2. Error (MSE)", "3. Loss Surface", 
    "4. Gradient Descent", "5. Learning Rate", "6. Robustness"
])

y_pred = m * X + b
current_mse = np.mean((y - y_pred)**2)

with tab1:
    st.markdown("**Concept:** We assume $y = mx + b$. Adjust $m$ and $b$ in the sidebar to fit the line.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name=f'Line: y={m}x+{b}', line=dict(color='red')))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(f"**Current MSE:** `{current_mse:.2f}`")
    st.markdown("**Concept:** Residuals are the vertical distances between the points and the line. We want to minimize the squared sum of these distances.")
    
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data'))
    fig_error.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Prediction', line=dict(color='red')))
    
    # Draw vertical residual lines
    for i in range(len(X)):
        fig_error.add_trace(go.Scatter(
            x=[X[i], X[i]], y=[y[i], y_pred[i]], 
            mode='lines', line=dict(color='gray', dash='dot'), showlegend=False
        ))
    st.plotly_chart(fig_error, use_container_width=True)

    with tab3:
        st.markdown("**Concept:** The Loss Surface shows how the error changes for every possible combination of $m$ and $b$.")
        
        # Generate a grid of m and b values
        m_grid = np.linspace(-5, 10, 50)
        b_grid = np.linspace(-50, 100, 50)
        M, B = np.meshgrid(m_grid, b_grid)
        
        # Calculate MSE for the entire grid
        Z = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                preds = M[i,j] * X + B[i,j]
                Z[i,j] = np.mean((y - preds)**2)
                
        fig_surface = go.Figure(data=[go.Surface(z=Z, x=M, y=B, colorscale='Viridis')])
        
        # Plot current position
        fig_surface.add_trace(go.Scatter3d(
            x=[m], y=[b], z=[current_mse],
            mode='markers', marker=dict(size=8, color='red'), name='Current Parameters'
        ))
        
        fig_surface.update_layout(scene=dict(xaxis_title='Slope (m)', yaxis_title='Intercept (b)', zaxis_title='MSE'))
        st.plotly_chart(fig_surface, use_container_width=True)

with tab4:
    st.markdown("**Concept:** Gradient descent takes steps down the loss surface to find the minimum.")
    
    if run_gd:
        gd_m, gd_b = m, b # Start from slider values
        history_m, history_b, history_loss = [gd_m], [gd_b], [current_mse]
        
        # Placeholder for animation
        plot_spot = st.empty() 
        
        for i in range(iterations):
            # Calculate gradients
            preds = gd_m * X + gd_b
            error = preds - y
            dm = (2/len(X)) * np.dot(error, X)
            db = (2/len(X)) * np.sum(error)
            
            # Update parameters
            gd_m -= learning_rate * dm
            gd_b -= learning_rate * db
            
            # Track history
            history_m.append(gd_m)
            history_b.append(gd_b)
            loss = np.mean((y - (gd_m * X + gd_b))**2)
            history_loss.append(loss)
            
            # (Optional) Update a contour plot dynamically in plot_spot
            # st.write(f"Step {i}: m={gd_m:.2f}, b={gd_b:.2f}, Loss={loss:.2f}")
            
        st.success("Optimization Complete!")
        st.line_chart(history_loss) # Simple loss curve