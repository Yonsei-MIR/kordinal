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

    def get_endpoint(self):
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
        # 요청 완료 후 성공 여부에 따라 request_counts 조정 가능
        # 여기서는 단순히 성공/실패와 무관하게 카운트는 이미 증가된 상태
        pass

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
