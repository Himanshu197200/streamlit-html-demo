import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
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


#### --- STEP 5: Tab 5 (Learning Rate Experiments) --- ####
with tab5:
    st.markdown("### The Impact of Step Size")
    st.markdown("**Concept:** The Learning Rate ($\\alpha$) controls how big of a step we take down the loss surface during Gradient Descent. If it's too small, learning is slow. If it's too big, the model overshoots and diverges.")
    
    st.markdown("Let's compare three different learning rates simultaneously, starting from the current $m$ and $b$ values in your sidebar.")
    
    # Use columns to let the user pick comparison rates
    col1, col2, col3 = st.columns(3)
    lr1 = col1.number_input("Learning Rate 1", value=0.0001, format="%f", step=0.0001)
    lr2 = col2.number_input("Learning Rate 2", value=0.0010, format="%f", step=0.0001)
    lr3 = col3.number_input("Learning Rate 3", value=0.0050, format="%f", step=0.0001)
    
    if st.button("Run Comparison", key="run_lr_comp"):
        # Helper function to run GD and return the loss history
        def run_gd_sim(lr_to_test):
            curr_m, curr_b = m, b # Start from sidebar values
            loss_history = []
            for _ in range(iterations): # Iterations from sidebar
                preds = curr_m * X + curr_b
                error = preds - y
                dm = (2/len(X)) * np.dot(error, X)
                db = (2/len(X)) * np.sum(error)
                
                curr_m -= lr_to_test * dm
                curr_b -= lr_to_test * db
                
                loss = np.mean((y - (curr_m * X + curr_b))**2)
                loss_history.append(loss)
            return loss_history
        
        # Run simulations
        loss_1 = run_gd_sim(lr1)
        loss_2 = run_gd_sim(lr2)
        loss_3 = run_gd_sim(lr3)
        
        # Plot the comparison
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(y=loss_1, mode='lines', name=f'LR = {lr1}', line=dict(width=3)))
        fig_lr.add_trace(go.Scatter(y=loss_2, mode='lines', name=f'LR = {lr2}', line=dict(width=3)))
        fig_lr.add_trace(go.Scatter(y=loss_3, mode='lines', name=f'LR = {lr3}', line=dict(width=3)))
        
        fig_lr.update_layout(
            title="Loss vs. Iterations", 
            xaxis_title="Iteration", 
            yaxis_title="Mean Squared Error (Loss)"
        )
        st.plotly_chart(fig_lr, use_container_width=True)
        
    st.info("💡 **What you should observe:** Try setting 'Learning Rate 3' to a high number (like `0.01`). You should see its line shoot straight up (divergence) because the steps are too large, while the smaller learning rates smoothly curve downward to minimize the error.")


#### --- STEP 6: Tab 6 (Noise & Robustness) --- ####
with tab6:
    st.markdown("### Sensitivity to Outliers")
    st.markdown("**Concept:** Linear regression tries to minimize *Mean Squared Error*. Because the errors are squared, massive outliers carry a huge penalty. This causes the best-fit line to skew heavily toward extreme data points.")
    
    st.markdown("Below, we use an algorithm to calculate the absolute mathematical **best fit** line for a perfectly clean dataset, and compare it to the best fit line for your current dataset.")
    
    # Generate clean baseline data for comparison
    X_clean, y_clean = generate_data(noise_level=0.0, add_outliers=False)
    
    # Reshape X arrays because Scikit-Learn expects 2D arrays
    X_sk = X.reshape(-1, 1)
    X_clean_sk = X_clean.reshape(-1, 1)
    
    # Fit exact mathematical models using scikit-learn
    model_clean = LinearRegression().fit(X_clean_sk, y_clean)
    model_current = LinearRegression().fit(X_sk, y)
    
    # Generate line predictions
    y_pred_clean = model_clean.predict(X_clean_sk)
    y_pred_current = model_current.predict(X_sk)
    
    # Build visual comparison
    fig_robust = go.Figure()
    
    # Plot current noisy/outlier data points
    fig_robust.add_trace(go.Scatter(
        x=X, y=y, mode='markers', name='Current Data', marker=dict(color='rgba(0, 100, 255, 0.6)')
    ))
    
    # Plot Baseline Line (Clean)
    fig_robust.add_trace(go.Scatter(
        x=X_clean, y=y_pred_clean, mode='lines', 
        name='Ideal Line (Clean Data)', line=dict(color='green', dash='dash', width=3)
    ))
    
    # Plot Adjusted Line (Current Data)
    fig_robust.add_trace(go.Scatter(
        x=X, y=y_pred_current, mode='lines', 
        name='Current Best Fit', line=dict(color='red', width=4)
    ))
    
    fig_robust.update_layout(title="Comparing Ideal Fit vs. Noisy Fit")
    st.plotly_chart(fig_robust, use_container_width=True)
    
    st.info("💡 **What you should observe:** Go to the sidebar and toggle **'Inject Outliers'** ON. Watch how drastically the solid Red line gets 'pulled' up or down by those 3 crazy data points compared to the dashed Green ideal line. This proves why standard Linear Regression isn't always robust to dirty real-world data.")