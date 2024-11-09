import numpy as np

# Beale function definition
def beale_function(x, y):
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

# Gradients of the Beale function
def beale_gradients(x, y):
    dx = 2 * (1.5 - x + x * y) * (y - 1) + 2 * (2.25 - x + x * y**2) * (y**2 - 1) + 2 * (2.625 - x + x * y**3) * (y**3 - 1)
    dy = 2 * (1.5 - x + x * y) * x + 2 * (2.25 - x + x * y**2) * (2 * x * y) + 2 * (2.625 - x + x * y**3) * (3 * x * y**2)
    return dx, dy

# ADAM optimizer implementation
def adam_optimizer(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=10000):
    x, y = 1.0, 1.0  # Initial guess
    m_x, m_y = 0.0, 0.0
    v_x, v_y = 0.0, 0.0
    t = 0

    for _ in range(max_iter):
        t += 1
        dx, dy = beale_gradients(x, y)
        
        # Update biased first moment estimate
        m_x = beta1 * m_x + (1 - beta1) * dx
        m_y = beta1 * m_y + (1 - beta1) * dy
        
        # Update biased second moment estimate
        v_x = beta2 * v_x + (1 - beta2) * (dx**2)
        v_y = beta2 * v_y + (1 - beta2) * (dy**2)
        
        # Compute bias-corrected first moment estimate
        m_x_hat = m_x / (1 - beta1**t)
        m_y_hat = m_y / (1 - beta1**t)
        
        # Compute bias-corrected second moment estimate
        v_x_hat = v_x / (1 - beta2**t)
        v_y_hat = v_y / (1 - beta2**t)
        
        # Update parameters
        x -= lr * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        y -= lr * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
        
        if np.sqrt(dx**2 + dy**2) < epsilon:  # Convergence criterion
            break

    return x, y

# Run the optimizer
x_opt, y_opt = adam_optimizer()
print(f"Optimized parameters: x = {x_opt}, y = {y_opt}")
