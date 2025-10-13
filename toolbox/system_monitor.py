import platform
import subprocess
import os
import psutil
import torch
from typing import Optional, Dict, Tuple, Union

NumericValue = Union[int, float]
MetricsDict = Dict[str, NumericValue]

class SystemMonitor:
    @staticmethod
    def get_nvidia_gpu_info() -> Tuple[str, MetricsDict, Optional[str]]:
        """Get NVIDIA GPU information and metrics for GPU 0."""
        metrics = {}
        gpu_name_from_torch = "NVIDIA GPU (name unavailable)"
        warning_message = None # To indicate if nvidia-smi failed and we're using PyTorch fallback

        try:
            gpu_name_from_torch = f"{torch.cuda.get_device_name(0)}"
        except Exception:
            # If even the name fails, nvidia-smi is highly likely to fail too.
            # Prepare basic PyTorch metrics as the ultimate fallback.
            metrics = {
                'memory_used_gb': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
                # Add placeholders for other metrics to maintain UI symmetry if nvidia-smi fails
                'memory_reserved_gb': 0.0,
                'temperature': 0.0,
                'utilization': 0.0
            }
            warning_message = "Could not get GPU name via PyTorch. nvidia-smi likely to fail or has failed. Displaying basic PyTorch memory (application-specific)."
            return gpu_name_from_torch, metrics, warning_message

        # Query for memory.used, memory.total, memory.reserved, temperature.gpu, utilization.gpu
        nvidia_smi_common_args = [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total,memory.reserved,temperature.gpu,utilization.gpu',
            '--format=csv,nounits,noheader'
        ]

        smi_output_str = None
        try:
            # Attempt 1: Query specific GPU 0
            smi_output_str = subprocess.check_output(
                nvidia_smi_common_args + ['--id=0'],
                encoding='utf-8', timeout=1.5, stderr=subprocess.PIPE
            )
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e1:
            # print(f"nvidia-smi with --id=0 failed: {type(e1).__name__}. Trying general query.")
            try:
                # Attempt 2: Query all GPUs and parse the first line
                smi_output_str = subprocess.check_output(
                    nvidia_smi_common_args, # Without --id=0
                    encoding='utf-8', timeout=1.5, stderr=subprocess.PIPE
                )
                if smi_output_str:
                    smi_output_str = smi_output_str.strip().split('\n')[0] # Take the first line
            except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e2:
                # print(f"nvidia-smi (general query) also failed: {type(e2).__name__}. Falling back to torch.cuda.")
                # Fallback to basic CUDA info from PyTorch, plus placeholders for UI
                metrics = {
                    'memory_used_gb': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
                    'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
                    'memory_reserved_gb': 0.0, # Placeholder
                    'temperature': 0.0,       # Placeholder
                    'utilization': 0.0        # Placeholder
                }
                warning_message = "nvidia-smi failed. GPU Memory Used is PyTorch specific (not total). Other GPU stats unavailable."
                return gpu_name_from_torch, metrics, warning_message

        if smi_output_str:
            parts = smi_output_str.strip().split(',')
            if len(parts) == 5: # memory.used, memory.total, memory.reserved, temp, util
                memory_used_mib, memory_total_mib, memory_reserved_mib, temp, util = map(float, parts)
                metrics = {
                    'memory_used_gb': memory_used_mib / 1024,
                    'memory_total_gb': memory_total_mib / 1024,
                    'memory_reserved_gb': memory_reserved_mib / 1024, # This is from nvidia-smi's memory.reserved
                    'temperature': temp,
                    'utilization': util
                }
            else:
                # print(f"Unexpected nvidia-smi output format: {smi_output_str}. Parts: {len(parts)}")
                warning_message = "nvidia-smi output format unexpected. Some GPU stats may be missing or inaccurate."
                # Fallback with placeholders to maintain UI structure
                metrics = {
                    'memory_used_gb': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0, # PyTorch fallback
                    'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0, # PyTorch fallback
                    'memory_reserved_gb': 0.0,
                    'temperature': 0.0,
                    'utilization': 0.0
                }
                if len(parts) >= 2: # Try to parse what we can if format is just partially off
                    try: metrics['memory_used_gb'] = float(parts[0]) / 1024
                    except: pass
                    try: metrics['memory_total_gb'] = float(parts[1]) / 1024
                    except: pass
        else: # Should have been caught by try-except, but as a final safety
            metrics = {
                'memory_used_gb': 0.0, 'memory_total_gb': 0.0, 'memory_reserved_gb': 0.0,
                'temperature': 0.0, 'utilization': 0.0
            }
            warning_message = "Failed to get any output from nvidia-smi."


        return gpu_name_from_torch, metrics, warning_message

    @staticmethod
    def get_mac_gpu_info() -> Tuple[str, MetricsDict, Optional[str]]: # Add warning return for consistency
        """Get Apple Silicon GPU information without requiring sudo."""
        metrics = {}
        warning_message = None
        try:
            memory = psutil.virtual_memory()
            metrics = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3), # This is system RAM, reported as "Unified Memory"
                'utilization': psutil.cpu_percent(),  # Still CPU usage as proxy
                # Placeholders for Mac to match NVIDIA's output structure for UI symmetry
                'memory_reserved_gb': 0.0, # N/A for unified memory in this context
                'temperature': 0.0 # Not easily available without sudo
            }
            if metrics['utilization'] == psutil.cpu_percent(): # Check if it's actually CPU util
                 warning_message = "Mac GPU Load is proxied by CPU Usage."

        except Exception as e:
            # print(f"Error getting Mac info: {e}")
            metrics = {
                'memory_total_gb': 0.0, 'memory_used_gb': 0.0, 'utilization': 0.0,
                'memory_reserved_gb': 0.0, 'temperature': 0.0
            }
            warning_message = "Could not retrieve Mac system info."
        
        return "Apple Silicon GPU", metrics, warning_message # Changed name for clarity

    @staticmethod
    def get_amd_gpu_info() -> Tuple[str, MetricsDict, Optional[str]]: # Add warning return
        """Get AMD GPU information."""
        metrics = { # Initialize with placeholders for all expected keys for UI symmetry
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'memory_reserved_gb': 0.0, # Typically N/A or not reported by rocm-smi in a 'reserved' sense
            'temperature': 0.0,
            'utilization': 0.0
        }
        warning_message = None
        source = "unknown"

        try:
            # Try rocm-smi first
            try:
                result = subprocess.check_output(['rocm-smi', '--showmeminfo', 'vram', '--showtemp', '--showuse'], encoding='utf-8', timeout=1.5, stderr=subprocess.PIPE)
                # Example rocm-smi output parsing (highly dependent on actual output format)
                # This needs to be robust or use a more structured output format like --json if rocm-smi supports it
                # For VRAM Used/Total:
                # GPU[0]		VRAM Usage: 2020M/16368M
                # For Temp:
                # GPU[0]		Temperature: 34c
                # For Util:
                # GPU[0]		GPU Use: 0%
                lines = result.strip().split('\n')
                for line in lines:
                    if line.startswith("GPU[0]"): # Assuming card 0
                        if "VRAM Usage:" in line:
                            mem_parts = line.split("VRAM Usage:")[1].strip().split('/')
                            metrics['memory_used_gb'] = float(mem_parts[0].replace('M', '')) / 1024
                            metrics['memory_total_gb'] = float(mem_parts[1].replace('M', '')) / 1024
                            source = "rocm-smi"
                        elif "Temperature:" in line:
                            metrics['temperature'] = float(line.split("Temperature:")[1].strip().replace('c', ''))
                            source = "rocm-smi"
                        elif "GPU Use:" in line:
                            metrics['utilization'] = float(line.split("GPU Use:")[1].strip().replace('%', ''))
                            source = "rocm-smi"
                if source != "rocm-smi": # if parsing failed or fields were missing
                    warning_message = "rocm-smi ran but output parsing failed."
            except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e_rocm:
                # print(f"rocm-smi failed: {e_rocm}. Trying sysfs.")
                warning_message = "rocm-smi not found or failed. "
                # Try sysfs as fallback on Linux
                if platform.system() == "Linux":
                    base_path = "/sys/class/drm/card0/device" # This assumes card0
                    sysfs_found_any = False
                    try:
                        with open(f"{base_path}/hwmon/hwmon0/temp1_input") as f: # Check for specific hwmon index
                            metrics['temperature'] = float(f.read().strip()) / 1000
                        sysfs_found_any = True
                    except (FileNotFoundError, PermissionError, ValueError): pass # Ignore if specific file not found
                    
                    try:
                        with open(f"{base_path}/mem_info_vram_total") as f:
                            metrics['memory_total_gb'] = int(f.read().strip()) / (1024**3) # Bytes to GiB
                        with open(f"{base_path}/mem_info_vram_used") as f:
                            metrics['memory_used_gb'] = int(f.read().strip()) / (1024**3) # Bytes to GiB
                        sysfs_found_any = True
                    except (FileNotFoundError, PermissionError, ValueError): pass
                            
                    try:
                        with open(f"{base_path}/gpu_busy_percent") as f:
                            metrics['utilization'] = float(f.read().strip())
                        sysfs_found_any = True
                    except (FileNotFoundError, PermissionError, ValueError): pass
                    
                    if sysfs_found_any:
                        source = "sysfs"
                        warning_message += "Using sysfs (may be incomplete)."
                    else:
                        warning_message += "sysfs also failed or provided no data."
                else:
                     warning_message += "Not on Linux, no sysfs fallback."
        
        except Exception as e_amd_main: # Catch-all for unforeseen issues in AMD block
            # print(f"General error in get_amd_gpu_info: {e_amd_main}")
            warning_message = (warning_message or "") + " Unexpected error in AMD GPU info gathering."
        
        return f"AMD GPU ({source})", metrics, warning_message

    @staticmethod
    def is_amd_gpu() -> bool: # No changes needed here
        try:
            # Check for rocm-smi first as it's more definitive
            rocm_smi_exists = False
            try:
                subprocess.check_call(['rocm-smi', '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5)
                rocm_smi_exists = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass # rocm-smi not found or errored
            
            if rocm_smi_exists:
                return True

            # Fallback to sysfs check if on Linux
            if platform.system() == "Linux" and os.path.exists('/sys/class/drm/card0/device/vendor'):
                with open('/sys/class/drm/card0/device/vendor', 'r') as f:
                    return '0x1002' in f.read() # AMD's PCI vendor ID
            return False
        except:
            return False

    @classmethod
    def get_system_info(cls) -> str:
        """Get detailed system status with support for different GPU types."""
        gpu_name_display: Optional[str] = None
        metrics: MetricsDict = {}
        gpu_warning: Optional[str] = None

        try:
            # Determine GPU type and get metrics
            if torch.cuda.is_available(): # Implies NVIDIA usually
                gpu_name_display, metrics, gpu_warning = cls.get_nvidia_gpu_info()
            elif platform.system() == "Darwin" and platform.processor() == "arm": # Apple Silicon
                gpu_name_display, metrics, gpu_warning = cls.get_mac_gpu_info()
            elif cls.is_amd_gpu(): # Check for AMD (works on Linux, might need refinement for Windows if not using PyTorch ROCm)
                gpu_name_display, metrics, gpu_warning = cls.get_amd_gpu_info()
            else: # No specific GPU detected by these primary checks
                # Could add a PyTorch ROCm check here if desired for AMD on Windows/Linux without rocm-smi
                # if hasattr(torch, 'rocm_is_available') and torch.rocm_is_available():
                # gpu_name_display = "AMD GPU (via PyTorch ROCm)"
                # metrics = { ... basic torch.rocm metrics ... }
                pass


            # Format GPU info based on available metrics
            if gpu_name_display:
                gpu_info_lines = [f"🎮 GPU: {gpu_name_display}"]
                
                # Standard memory reporting
                if 'memory_used_gb' in metrics and 'memory_total_gb' in metrics:
                    mem_label = "GPU Memory"
                    if platform.system() == "Darwin" and platform.processor() == "arm":
                        mem_label = "Unified Memory" # For Apple Silicon
                    
                    gpu_info_lines.append(
                        f"📊 {mem_label}: {metrics.get('memory_used_gb', 0.0):.1f}GB / {metrics.get('memory_total_gb', 0.0):.1f}GB"
                    )

                # VRAM Reserved (NVIDIA specific from nvidia-smi, or placeholder)
                # if 'memory_reserved_gb' in metrics and torch.cuda.is_available() and not (platform.system() == "Darwin"): # Show for NVIDIA, not Mac
                    # gpu_info_lines.append(f"💾 VRAM Reserved: {metrics.get('memory_reserved_gb', 0.0):.1f}GB")
                
                if 'temperature' in metrics and metrics.get('temperature', 0.0) > 0: # Only show if temp is valid
                    gpu_info_lines.append(f"🌡️ GPU Temp: {metrics.get('temperature', 0.0):.0f}°C")
                
                if 'utilization' in metrics:
                    gpu_info_lines.append(f"⚡ GPU Load: {metrics.get('utilization', 0.0):.0f}%")
                
                if gpu_warning: # Display any warning from the GPU info functions
                    gpu_info_lines.append(f"⚠️ {gpu_warning}")
                    
                gpu_section = "\n".join(gpu_info_lines) + "\n"
            else:
                gpu_section = "🎮 GPU: No dedicated GPU detected or supported\n"
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False) # Physical cores
            cpu_threads = psutil.cpu_count(logical=True) # Logical processors
            cpu_info = f"💻 CPU: {cpu_count or 'N/A'} Cores, {cpu_threads or 'N/A'} Threads\n"
            cpu_usage = f"⚡ CPU Usage: {psutil.cpu_percent()}%\n"
            
            # Get RAM info
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            ram_info = f"🎯 System RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent}%)"
            
            return f"{gpu_section}{cpu_info}{cpu_usage}{ram_info}"
            
        except Exception as e:
            # print(f"Overall error in get_system_info: {e}")
            # import traceback; print(traceback.format_exc())
            return f"Error collecting system info: {str(e)}"