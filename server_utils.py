import threading
import time
import requests

class EndpointSelector:
    def __init__(self, endpoints, health_url="http://{host}:{port}/health"):
        """
        Initialize the EndpointSelector with a list of endpoints.
        Each endpoint is a tuple of (host, port).
        """
        self.endpoints = endpoints  # Store endpoints
        self.lock = threading.Lock()  # Thread-safe access
        self.index = 0  # Start index for round-robin
        self.health_url = health_url

    def check_health(self, host, port):
        """
        Check the health of the given endpoint.
        Returns True if healthy, False otherwise.
        """
        health_url = self.health_url.format(host=host, port=port)
        try:
            response = requests.get(health_url)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_next_endpoint(self):
        """
        Return the next endpoint in the round-robin order.
        """
        with self.lock:
            endpoint = self.endpoints[self.index]
            self.index = (self.index + 1) % len(self.endpoints)
            return endpoint, self.index
    
    def available(self) -> tuple[str, int]:
        """
        Return a healthy endpoint, rotating the order of endpoints.
        This function only executes when explicitly called.
        """
        # return self.get_next_endpoint()
        for _ in range(len(self.endpoints)):
            endpoint, idx = self.get_next_endpoint()  # Get the next endpoint
            # return endpoint
            if isinstance(endpoint, str):
                host, port = endpoint.split(":")
            elif isinstance(endpoint, tuple):
                host, port = endpoint
            elif isinstance(endpoint, dict):
                host = endpoint["host"]
                port = endpoint["port"]
            else:
                raise ValueError(f"Invalid endpoint type. Must be str, tuple, or dict. But got {endpoint}")
            
            if self.check_health(host, port):  # Check health
                return endpoint, idx  # Return the healthy endpoint
            time.sleep(0.01)

        # If no endpoint is healthy
        raise Exception("No healthy endpoints available!")

    def success(self, idx):
        """
        Mark the given endpoint as successful.
        """
        pass