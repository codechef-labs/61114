import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # Generate 100 random x values
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5  # Generate y values with some noise

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared score
r2 = r2_score(y, y_pred)

# Create the visualization
plt.figure(figsize=(10, 6))

# Plot the scatter points
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')

# Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')

# Add labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')

# Add model information
equation = f'y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}x'
plt.text(0.05, 0.95, f'Equation: {equation}\nR² = {r2:.4f}', 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add legend
plt.legend()

# Display grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Print the model parameters
print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
print(f"Slope (β₁): {model.coef_[0][0]:.4f}")
print(f"R-squared: {r2:.4f}")
