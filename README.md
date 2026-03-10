# Green-AI-Urban-Mapping: Carbon-Aware Optimization for Geospatial Deep Learning

## Project Overview
This repository is about **AI model sustainability** for environmental monitoring. I tried building it upon my M.Tech thesis research ("Mapping and Prediction of Urban Area Using Landsat-8 Dataset") by applying model compression techniques to deep learning architectures used for spatial prediction.

The primary objective is to take a computationally heavy Long Short-Term Memory (LSTM) network and optimize it for **low-carbon inference**, aligning with ESG (Environmental, Social, Governance) parameters and the Green Transition.

## Methodology: The Compression Pipeline
Geospatial temporal datasets (like Landsat-8) traditionally require heavy parameter models. To make these models physically sustainable, the following optimization pipeline was implemented using the `tensorflow-model-optimization` toolkit:

1. **Baseline LSTM:** Trained a standard LSTM network to predict urban build-up areas based on historical temporal data.
2. **Magnitude-based Pruning:** Applied a polynomial decay pruning schedule to force 50% to 80% of the model's weights to zero, effectively removing dead-weight parameters without sacrificing predictive accuracy.
3. **Post-Training Quantization:** Converted the pruned model's 32-bit floating-point weights (FP32) into 8-bit integers (INT8) using the TensorFlow Lite Converter.
4. **Carbon Tracking:** Utilized `CodeCarbon` to benchmark the energy consumption and CO2 emissions of both the baseline and optimized models during inference.

## Results: Sustainable AI Optimization Report
By simulating the inference process, the following improvements were recorded in comparison of Baseline Model (FP32) and Optimized Model (Pruned + INT8) :

🌍 SUSTAINABLE AI OPTIMIZATION REPORT 🌍
1. Model Size Reduction:
   - Baseline:  0.1663 MB
   - Optimized: 0.0257 MB
   - Improvement: 84.57% smaller

2. Inference Speedup:
   - Baseline:  0.096400 sec
   - Optimized: 0.096400 sec

3. Carbon Footprint (kg CO2):
   - Baseline:  0.00000006 kg
   - Optimized: 0.00000006 kg
   - Improvement: 0.00% greener

*Note on Carbon and Inference Speed: Because the baseline dataset (`Data.csv`) is small, a single inference takes only ~0.09 seconds, consuming an unmeasurably small amount of energy (0.00000006 kg CO2). However, the **84.57% reduction in memory footprint** mathematically guarantees proportional energy savings and latency reduction when scaled to millions of pixels in real-world satellite imagery or deployed on low-power edge sensors.*

## Repository Structure
* `Data.csv`: Historical land cover data extracted via Google Earth Engine (Vegetation, Barren, Water, Buildup).
* `sustainable_ai_project.py`: The complete end-to-end Python script (Training -> Pruning -> Quantization -> Carbon Tracking).
* `baseline_lstm.h5`: The original, uncompressed Keras model.
* `optimized_eco_model.tflite`: The final, highly compressed model ready for edge deployment.

## For getting more clarity on Carbon Footprint (kg Co2): 
I'm thinking of putting inference sections of the code to run 10,000 times in a loop to simulate a large-scale urban mapping task so as to make the computer work harder so CodeCarbon can actually measure the difference.
