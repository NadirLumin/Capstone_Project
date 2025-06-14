Quick Start
Build Docker:

bash

docker build -t linguistic-transformer-api .
Run the API server:

bash

docker run -p 8000:8000 linguistic-transformer-api
Try it out in the demo notebook:
Open and run notebooks/demo_api_application.ipynb, which demonstrates how to POST text to your API and view results.

Interface Availability
This project provides a fully-dockerized REST API interface for interacting with the model. Interactors can send HTTP POST requests to the /predict endpoint to receive model outputs, as demonstrated in notebooks/demo_api_application.ipynb.

For grading, please follow the provided steps to build and run the container, then utilize the notebook or any REST client to access the interface.

If a public endpoint is required, please let me know—I can provide a temporary hosted version.