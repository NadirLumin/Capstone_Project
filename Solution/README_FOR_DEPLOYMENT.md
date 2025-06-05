Quick Start
1. Build Docker:

bash

docker build -t linguistic-transformer-api .
2. Run the API server:

bash

docker run -p 8000:8000 linguistic-transformer-api
3. Try it out in the demo notebook:

Open and run notebooks/demo_api_application.ipynb, which demonstrates how to POST text to your API and view results.

Example: Final Quick README Section
To run this app:

Build the container:

nginx

docker build -t linguistic-transformer-api .
Start the server:

arduino

docker run -p 8000:8000 linguistic-transformer-api
Open and run notebooks/demo_api_application.ipynb for an interactive demo (simple press run on the first cell).