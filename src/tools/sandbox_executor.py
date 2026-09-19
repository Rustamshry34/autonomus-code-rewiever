import docker
import base64
import time
from docker.errors import ContainerError, APIError, ImageNotFound
from typing import Any
from langchain_core.tools import tool
from src.agent.state import TestResult
from config.settings import settings
import structlog

logger = structlog.get_logger()


# ============================================================================
# Sandbox Executor
# ============================================================================

class SandboxExecutor:
    """
    Docker-based sandbox executor.
    Kodu izole konteynırlarda güvenli bir şekilde çalıştırır.
    
    Güvenlik önlemleri:
    - Network izolasyonu (network_disabled=True)
    - Memory/CPU limitleri
    - Read-only filesystem
    - Timeout kontrolü
    - no-new-privileges security opt
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.image_name = settings.sandbox_image
        self.timeout = settings.sandbox_timeout
        
        # Container güvenlik ayarları
        self.security_config = {
            "mem_limit": "256m",
            "cpu_period": 100000,
            "cpu_quota": 50000,  # %50 CPU
            "network_disabled": True,
            "read_only": True,
            "security_opt": ["no-new-privileges"],
            "detach": False,
            "stdout": True,
            "stderr": True,
        }
    
    def execute_code(
        self,
        code: str,
        file_name: str = "sandbox_code.py",
        timeout: int | None = None
    ) -> TestResult:
        """
        Python kodunu sandbox'ta çalıştırır.
        
        Args:
            code: Çalıştırılacak Python kodu
            file_name: Kod dosyası adı (loglama için)
            timeout: Override timeout (saniye)
            
        Returns:
            TestResult dataclass instance
        """
        start_time = time.time()
        exec_timeout = timeout or self.timeout
        
        # Kodu base64 encode et (dosya sistemi gerektirmeden geçirmek için)
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
        
        # Container command: base64 decode edip python'a pipe'la
        command = f"echo '{encoded_code}' | base64 -d | python3 -"
        
        try:
            logger.info(
                "Executing code in sandbox",
                file_name=file_name,
                code_length=len(code)
            )
            
            result = self.docker_client.containers.run(
                image=self.image_name,
                command=["sh", "-c", command],
                timeout=exec_timeout,
                **self.security_config
            )
            
            execution_time = time.time() - start_time
            
            logger.info(
                "Code execution completed",
                file_name=file_name,
                execution_time=execution_time,
                success=True
            )
            
            return TestResult(
                test_name=file_name,
                passed=True,
                output=result.decode('utf-8'),
                execution_time=execution_time
            )
            
        except ContainerError as e:
            execution_time = time.time() - start_time
            
            stdout = e.stdout.decode('utf-8') if e.stdout else ""
            stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
            
            logger.warning(
                "Code execution failed",
                file_name=file_name,
                execution_time=execution_time,
                error=stderr
            )
            
            return TestResult(
                test_name=file_name,
                passed=False,
                output=stdout,
                error_message=stderr,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "Sandbox execution error",
                file_name=file_name,
                error=str(e)
            )
            
            return TestResult(
                test_name=file_name,
                passed=False,
                output="",
                error_message=f"Sandbox error: {str(e)}",
                execution_time=execution_time
            )
    
    def run_pytest(
        self,
        test_code: str,
        file_name: str = "test_sandbox.py",
        timeout: int | None = None
    ) -> TestResult:
        """
        Pytest testlerini sandbox'ta çalıştırır.
        
        Args:
            test_code: Test kodu
            file_name: Test dosyası adı
            timeout: Override timeout (saniye)
            
        Returns:
            TestResult dataclass instance
        """
        start_time = time.time()
        exec_timeout = timeout or self.timeout
        
        # Test kodunu base64 encode et
        encoded_test = base64.b64encode(test_code.encode('utf-8')).decode('utf-8')
        
        # Pytest command
        command = f"echo '{encoded_test}' | base64 -d > /tmp/{file_name} && pytest /tmp/{file_name} -v"
        
        try:
            logger.info(
                "Running pytest in sandbox",
                file_name=file_name,
                code_length=len(test_code)
            )
            
            result = self.docker_client.containers.run(
                image=self.image_name,
                command=["sh", "-c", command],
                timeout=exec_timeout,
                **self.security_config
            )
            
            execution_time = time.time() - start_time
            
            logger.info(
                "Pytest execution completed",
                file_name=file_name,
                execution_time=execution_time,
                success=True
            )
            
            return TestResult(
                test_name=file_name,
                passed=True,
                output=result.decode('utf-8'),
                execution_time=execution_time
            )
            
        except ContainerError as e:
            execution_time = time.time() - start_time
            
            stdout = e.stdout.decode('utf-8') if e.stdout else ""
            stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
            
            # Pytest exit codes: 1 = test failed, 2 = test error, 5 = no tests
            logger.warning(
                "Pytest execution failed",
                file_name=file_name,
                execution_time=execution_time,
                error=stderr
            )
            
            return TestResult(
                test_name=file_name,
                passed=False,
                output=stdout,
                error_message=stderr,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "Pytest sandbox error",
                file_name=file_name,
                error=str(e)
            )
            
            return TestResult(
                test_name=file_name,
                passed=False,
                output="",
                error_message=f"Sandbox error: {str(e)}",
                execution_time=execution_time
            )
    
    def verify_sandbox_image(self) -> bool:
        """
        Sandbox Docker imajının mevcut olup olmadığını kontrol eder.
        
        Returns:
            True ise imaj mevcut
        """
        try:
            self.docker_client.images.get(self.image_name)
            logger.info("Sandbox image found", image_name=self.image_name)
            return True
        except ImageNotFound:
            logger.error("Sandbox image not found", image_name=self.image_name)
            return False
        except Exception as e:
            logger.error("Error checking sandbox image", error=str(e))
            return False
    
    def build_sandbox_image(self) -> bool:
        """
        Sandbox Docker imajını build eder.
        
        Returns:
            True ise build başarılı
        """
        try:
            logger.info("Building sandbox image", image_name=self.image_name)
            
            self.docker_client.images.build(
                path="./sandbox",
                tag=self.image_name,
                rm=True,
                forcerm=True
            )
            
            logger.info("Sandbox image built successfully", image_name=self.image_name)
            return True
            
        except Exception as e:
            logger.error("Failed to build sandbox image", error=str(e))
            return False


# ============================================================================
# Singleton Instance
# ============================================================================

sandbox_executor = SandboxExecutor()


# ============================================================================
# LangChain Tool Wrapper'ları
# ============================================================================

@tool
def execute_code_in_sandbox(code: str, file_name: str = "sandbox_code.py") -> dict[str, Any]:
    """
    Executes Python code in an isolated Docker sandbox.
    The sandbox has network disabled, read-only filesystem, and resource limits.
    
    Args:
        code: Python code to execute
        file_name: Name for the code file (for logging)
        
    Returns:
        Dictionary with execution results
    """
    result = sandbox_executor.execute_code(code, file_name)
    
    return {
        "test_name": result.test_name,
        "passed": result.passed,
        "output": result.output,
        "error_message": result.error_message,
        "execution_time": result.execution_time
    }


@tool
def run_pytest_in_sandbox(test_code: str, file_name: str = "test_sandbox.py") -> dict[str, Any]:
    """
    Runs pytest tests in an isolated Docker sandbox.
    
    Args:
        test_code: Pytest test code to run
        file_name: Name for the test file (for logging)
        
    Returns:
        Dictionary with test results
    """
    result = sandbox_executor.run_pytest(test_code, file_name)
    
    return {
        "test_name": result.test_name,
        "passed": result.passed,
        "output": result.output,
        "error_message": result.error_message,
        "execution_time": result.execution_time
    }


@tool
def verify_sandbox_available() -> dict[str, Any]:
    """
    Verifies that the sandbox Docker image is available and ready.
    
    Returns:
        Dictionary with availability status
    """
    available = sandbox_executor.verify_sandbox_image()
    
    return {
        "available": available,
        "image_name": sandbox_executor.image_name
    }


@tool
def setup_sandbox_if_needed() -> dict[str, Any]:
    """
    Checks if sandbox image exists, builds it if not.
    
    Returns:
        Dictionary with setup status
    """
    if sandbox_executor.verify_sandbox_image():
        return {
            "status": "ready",
            "message": "Sandbox image already exists"
        }
    
    # Build et
    if sandbox_executor.build_sandbox_image():
        return {
            "status": "built",
            "message": "Sandbox image built successfully"
        }
    else:
        return {
            "status": "error",
            "message": "Failed to build sandbox image"
        }
