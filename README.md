# Marine Intelligence Guard: AI-Enabled Edge System

Marine Intelligence Guard is an end-to-end, downstream intelligence system designed to solve the "Satellite-to-Ground Bottleneck."

By simulating onboard edge intelligence, this system processes high-volume Synthetic Aperture Radar (SAR) imagery to detect maritime vessels in real-time, transmitting only actionable metadata to ground stations.

Key Engineering Impact:

* 99% Data Reduction: Successfully reduced downlink payload by transmitting JSON metadata instead of raw, high-resolution imagery.
* Production-Ready API: Developed a modular FastAPI inference engine for seamless integration into satellite ground-segment workflows.
* Real-Time Latency Mitigation: Addresses the critical 100+ GB/day bandwidth constraint inherent in modern SAR constellations.

# Software Architecture & Systems Design

Technical StackBackend: 
* FastAPI (Inference Engine) for high-performance, asynchronous processing.
* Frontend: Streamlit-based Ground Control Dashboard for real-time visualization.
* Image Processing: OpenCV for pre-processing and Base64-encoded visualization layers.
* Deployment: Designed for ONNX format compatibility to ensure high-speed execution on Edge devices.

# Computer Vision Pipeline
Detection Workflow
* Ingestion: Satellite SAR image capture (simulated 10–50 MB per image). 
* Inference: YOLOv8 detects ships even in high-noise environments. 
* Extraction: The system extracts bounding box coordinates and confidence scores.
* Transmission: Only a lightweight JSON payload is downlinked to the ground station.

# Space-Tech Applications
* National Security: Unauthorized entry detection and coastal monitoring.
* Environmental Monitoring: Surface mining expansion and flood impact assessment.
* Logistics: Port congestion monitoring and risk estimation for shipping corridors.

# Limitations & Future Roadmap
* Geospatial Precision: Currently focuses on bounding box detection; future iterations will integrate rasterio for precise orbit-based pixel-to-Lat/Long mapping.
* Temporal Analytics: Plans to implement multi-temporal change detection for vessel trajectory estimation and behavioral anomaly modeling.
* Environmental Robustness: Further training is required for extremely high sea clutter and heavy storm conditions.

# Author: Himanshu Raj
