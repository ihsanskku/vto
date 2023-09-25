import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras


# Prepare the data for LSTM
def prepare_data(trajectory, seq_length):
    X, y = [], []
    for i in range(len(trajectory) - seq_length):
        X.append(trajectory[i:i + seq_length])
        y.append(trajectory[i + seq_length])
    return np.array(X), np.array(y)


# LSTM Model Parameters
seq_length = 5
num_units = 50
batch_size = 1
epochs = 100

# Simulated vehicle trajectories
num_time_steps = 20
num_vehicles = 5

# Generate random starting points for each vehicle
start_points = np.random.randint(0, 100, size=num_vehicles)

# Generate random initial velocities for each vehicle
initial_velocities = np.random.uniform(2, 8, size=num_vehicles)

# Landmarks and their speed control effect
landmarks = {7: 0.7, 12: 0.8, 18: 0.9}

# Generate time steps
time_steps = np.arange(num_time_steps)

# Generate vehicle trajectories with varying speeds and landmarks
trajectories = []
for i in range(num_vehicles):
    trajectory = [start_points[i]]
    velocity = initial_velocities[i]
    for t in range(1, num_time_steps):
        speed_control = 1.0
        if t in landmarks:
            speed_control = landmarks[t]

        # Update velocity and consider noise
        velocity = velocity * speed_control + np.random.normal(0, 0.5)
        position = trajectory[-1] + velocity

        trajectory.append(position)

    trajectories.append(np.array(trajectory))

# Store predictions
predictions = []

for i, trajectory in enumerate(trajectories):
    # Data preparation
    X, y = prepare_data(trajectory, seq_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units=num_units, input_shape=(X.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=num_units))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Make predictions
    last_sequence = np.reshape(trajectory[-seq_length:], (1, seq_length, 1))
    next_point_pred = model.predict(last_sequence)[0][0]
    predictions.append(next_point_pred)

# Create a clean and informative plot
plt.figure(figsize=(12, 6))

# Define line styles for each vehicle
line_styles = ['-', '--', '-.', ':', '-']

for i, trajectory in enumerate(trajectories):
    plt.plot(time_steps, trajectory, label=f'Vehicle {i + 1}', linestyle=line_styles[i], linewidth=2)
    plt.scatter(num_time_steps, predictions[i], color=plt.gca().lines[-1].get_color(), s=70, zorder=10,
                label=f'Predicted {i + 1}')

plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Location', fontsize=12)
plt.title('Simulated Vehicle Trajectories with Realistic Dynamics and LSTM Predictions', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.5)
plt.show()
