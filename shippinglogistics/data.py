# 1. Generate a date range from 1970 to 2024
dates = pd.date_range(start="1970-01-01", periods=55, freq="Y")

# 2. Create synthetic logistics data
data = {
    "year": dates.year,

    # Simulate inventory turnover with a sinusoidal trend and some noise
    "inventory_turnover": np.round(np.abs(np.sin(np.linspace(0, 10, 55)) + np.random.normal(0, 0.2, 55)), 2),

    # Generate random on-time delivery percentages (between 85% and 99%)
    "on_time_in_full": np.round(np.random.uniform(85, 99, 55), 1),

    # Simulate lead time values in days (between 2 and 10 days)
    "lead_time": np.round(np.random.uniform(2, 10, 55), 1),

    # Generate transportation costs in dollars (ranging from $1000 to $5000)
    "transport_cost": np.round(np.random.uniform(1000, 5000, 55), 2)
}

# 3. Create a DataFrame
df = pd.DataFrame(data)