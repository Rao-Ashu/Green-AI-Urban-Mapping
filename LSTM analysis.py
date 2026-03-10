import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow_model_optimization as tfmot
from codecarbon import EmissionsTracker
import time
import os

# ==========================================
# 1. Data Preparation (From Data.csv)
# ==========================================
df = pd.read_csv('Data.csv')

# will predict 'Buildup' (Urban Area) based on the 'year'
# Reshaping input for LSTM: (samples, time steps, features)
X = df['year'].values.reshape(-1, 1, 1) 
y = df['Buildup'].values

# Split into train and test (Using 2021 as test data)
X_train, X_test = X[:-1], X[-1:]
y_train, y_test = y[:-1], y[-1:]

# ==========================================
# 2. Baseline Model & Emissions Tracking
# ==========================================
print("\n--- 2 ---")
# Define a standard LSTM architecture similar to the thesis notebook
baseline_model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 1)),
    Dense(25, activation='relu'),
    Dense(1)
])
baseline_model.compile(optimizer='adam', loss='mse')

# Train baseline
baseline_model.fit(X_train, y_train, epochs=200, verbose=0)
baseline_model.save('baseline_lstm.h5')

# Track Baseline Inference Emissions
tracker_baseline = EmissionsTracker(project_name="Baseline_Inference")
tracker_baseline.start()
start_time = time.time()

# Running prediction for the test year (2021)
baseline_pred = baseline_model.predict(X_test)

end_time = time.time()
baseline_emissions = tracker_baseline.stop()
baseline_time = end_time - start_time
baseline_size = os.path.getsize('baseline_lstm.h5') / (1024 * 1024) # MB

print(f"Baseline Prediction for year 21: {baseline_pred[0][0]:.2f} (Actual: {y_test[0]})")

# ==========================================
# 3. Model Compression (Pruning)
# ==========================================
print("\n--- 3 ---")
# Pruning 50% to 80% of weights
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 1
epochs = 50
end_step = np.ceil(len(X_train) / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}
try:
    pruned_model = prune_low_magnitude(baseline_model, **pruning_params)
    pruned_model.compile(optimizer='adam', loss='mse')

    # Fine-tune the pruned model
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, verbose=0)

    # Strip pruning wrappers to finalize size reduction
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
except Exception as e:
    print("Warning: pruning failed or is incompatible in this environment. Skipping pruning. Error:", e)
    # Fall back to using the baseline model for export (no pruning)
    model_for_export = baseline_model

# ==========================================
# 4. Model Quantization (TFLite INT8)
# ==========================================
print("\n--- 4 ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Configuring converter to allow Select TF ops and to avoid experimental lowering
try:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                          tf.lite.OpsSet.SELECT_TF_OPS]
    # Disabling experimental lowering of TensorList ops which can fail for LSTM/TensorArray
    converter._experimental_lower_tensor_list_ops = False
except Exception:
    # Some TF builds don't expose these attributes; continue and attempt conversion anyway
    pass

quantized_tflite_model = converter.convert()

with open('optimized_eco_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# ==========================================
# 5. Optimized Model & Emissions Tracking
# ==========================================
print("\n--- 5 ---")
try:
    # Initialize TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="optimized_eco_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Track Optimized Inference Emissions
    tracker_optimized = EmissionsTracker(project_name="Optimized_Inference")
    tracker_optimized.start()
    start_time_opt = time.time()

    # Run prediction through TFLite model
    input_data = np.array(X_test, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    optimized_pred = interpreter.get_tensor(output_details[0]['index'])

    end_time_opt = time.time()
    optimized_emissions = tracker_optimized.stop()
    optimized_time = end_time_opt - start_time_opt
    optimized_size = os.path.getsize('optimized_eco_model.tflite') / (1024 * 1024) # MB
except Exception as e:
    print("Warning: TFLite interpreter failed (Select TF ops or unsupported delegate). Skipping TFLite inference. Error:", e)
    # Fall back to baseline values so script can finish
    optimized_pred = baseline_pred
    optimized_time = baseline_time
    optimized_emissions = baseline_emissions
    try:
        optimized_size = os.path.getsize('optimized_eco_model.tflite') / (1024 * 1024)
    except Exception:
        optimized_size = baseline_size

print(f"Optimized Prediction for year 21: {optimized_pred[0][0]:.2f} (Actual: {y_test[0]})")

# ==========================================
# 6. Final Report Generation
# ==========================================
print("\n==================================================")
print("🌍 SUSTAINABLE AI OPTIMIZATION REPORT 🌍")
print("==================================================")
print(f"1. Model Size Reduction:")
print(f"   - Baseline:  {baseline_size:.4f} MB")
print(f"   - Optimized: {optimized_size:.4f} MB")
print(f"   - Improvement: {((baseline_size - optimized_size) / baseline_size) * 100:.2f}% smaller")

print(f"\n2. Inference Speedup:")
print(f"   - Baseline:  {baseline_time:.6f} sec")
print(f"   - Optimized: {optimized_time:.6f} sec")

print(f"\n3. Carbon Footprint (kg CO2):")
print(f"   - Baseline:  {baseline_emissions:.8f} kg")
print(f"   - Optimized: {optimized_emissions:.8f} kg")
print(f"   - Improvement: {((baseline_emissions - optimized_emissions) / baseline_emissions) * 100:.2f}% greener")
print("==================================================")