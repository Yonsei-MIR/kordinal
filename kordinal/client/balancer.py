from abc import ABC, abstractmethod
import itertools
import time
import random
import threading

class LoadBalancer(ABC):
    @abstractmethod
    def get_endpoint(self):
        pass

    @abstractmethod
    def update_metrics(self, endpoint, response_time, success=True):
        pass

class IndexBasedLoadBalancer(LoadBalancer):
    def __init__(self, endpoints: list):
        # endpoints: [{"host": "...", "port": 1234}, ...]
        self.endpoints = endpoints
        self.index = 0
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.endpoints)

    def __str__(self):
        return f"{self.__class__.__name__}({self.endpoints})"
    
    def __repr__(self) -> str:
        return str(self)

    def remove(self, endpoint):
        with self.lock:
            if endpoint in self.endpoints:
                self.endpoints.remove(endpoint)
                # Adjust index if it goes out of bounds
                if self.index >= len(self.endpoints):
                    self.index = 0
                return True  # Successfully removed
            return False  # Endpoint not found

    def get_endpoint(self):
        with self.lock:
            if not self.endpoints:
                raise RuntimeError("No endpoints available in the load balancer.")
            return self.endpoints[self.index]

    def update_metrics(self, endpoint, response_time, success=True):
        pass

class RoundRobinLoadBalancer(IndexBasedLoadBalancer):
    def __init__(self, endpoints):
        super().__init__(endpoints)
        
    def get_endpoint(self):
        with self.lock:
            endpoint = self.endpoints[self.index]
            self.index = (self.index + 1) % len(self.endpoints)
        return endpoint

class LeastRequestsLoadBalancer(IndexBasedLoadBalancer):
    def __init__(self, endpoints):
        super().__init__(endpoints)
        self.request_counts = {i:0 for i in range(len(endpoints))}

    def get_endpoint(self):
        with self.lock:
            min_idx = min(self.request_counts, key=self.request_counts.get)
            self.request_counts[min_idx] += 1
        return self.endpoints[min_idx]

    def update_metrics(self, endpoint, response_time, success=True):
        # 요청 완료 후 요청 카운트를 감소시킴
        with self.lock:
            try:
                idx = self.endpoints.index(endpoint)
                self.request_counts[idx] = max(0, self.request_counts[idx] - 1)  # 카운트 감소 (최소 0)
            except ValueError:
                pass  # 엔드포인트가 이미 제거된 경우

class FastestResponseLoadBalancer(IndexBasedLoadBalancer):
    def __init__(self, endpoints):
        super().__init__(endpoints)
        # 평균 응답속도를 추적하기 위한 데이터
        self.response_times = {i: 1.0 for i in range(len(endpoints))}

    def get_endpoint(self):
        with self.lock:
            # 평균 응답 시간이 가장 빠른 endpoint 선택
            min_idx = min(self.response_times, key=self.response_times.get)
        return self.endpoints[min_idx]

    def update_metrics(self, endpoint, response_time, success=True):
        with self.lock:
            idx = self.endpoints.index(endpoint)
            # 지수 가중 이동 평균 등으로 응답시간 갱신 가능
            alpha = 0.2
            self.response_times[idx] = alpha * response_time + (1 - alpha)*self.response_times[idx]
