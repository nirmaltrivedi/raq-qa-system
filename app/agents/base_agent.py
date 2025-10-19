from abc import ABC, abstractmethod
from typing import Dict, Any
from app.core.logging import app_logger as logger


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        
        pass

    def log_execution(self, action: str, details: str = ""):
        
        log_msg = f"[{self.name}] {action}"
        if details:
            log_msg += f": {details}"
        logger.info(log_msg)

    def log_error(self, error: str):
        
        logger.error(f"[{self.name}] Error: {error}")
