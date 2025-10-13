import torch
import logging
import gc

class ModelManager:
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.active_model_name = None
        # Keep basicConfig in case we need to re-enable logging for debugging.
        logging.basicConfig(level=logging.INFO)

    def add(self, name, model):
        self.models[name] = model.cpu()
        # logging.info(f"Model '{name}' registered with ModelManager on CPU.")

    def get(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in RAM.")

        if self.active_model_name and self.active_model_name != name:
            # logging.info(f"[MEM] Offloading '{self.active_model_name}' from GPU...")
            self.models[self.active_model_name].to('cpu')
            torch.cuda.synchronize()

        model = self.models[name]
        
        # Check the device of the *first parameter* to see if the model is on the GPU.
        # This is more reliable than checking a custom .device attribute.
        if next(model.parameters()).device.type != 'cpu':
            # logging.info(f"[MEM] Model '{name}' is already on the GPU.")
            pass
        else:
             # logging.info(f"[MEM] Loading '{name}' to GPU...")
             model.to(self.device)
             torch.cuda.synchronize()

        self.active_model_name = name
        torch.cuda.empty_cache()
        return model

    def clear_gpu(self):
        """Offloads the currently active model from the GPU."""
        if self.active_model_name:
            # logging.info(f"[MEM] Clearing '{self.active_model_name}' from GPU...")
            if self.active_model_name in self.models:
                self.models[self.active_model_name].cpu()
                torch.cuda.synchronize()
            self.active_model_name = None
            torch.cuda.empty_cache()
            gc.collect()

    def offload_all_models(self):
        """Iterates through all registered models and moves them to CPU."""
        if not self.models:
            return

        print("[MEM] Offloading all models from GPU to CPU...")
        for name, model in self.models.items():
            # Check if the model has parameters and if they are not already on the CPU
            if hasattr(model, 'parameters') and next(model.parameters(), None) is not None:
                if next(model.parameters()).device.type != 'cpu':
                    model.cpu()
        
        torch.cuda.synchronize()
        self.active_model_name = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[MEM] All models offloaded. VRAM cleared.")

    def clear_all_references(self):
        """Clears all model references from the manager with aggressive cleanup."""
        print("[MEM] Clearing all model references from ModelManager.")
        
        # First move everything to CPU if not already there
        for name, model in list(self.models.items()):
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                # Explicitly delete the model reference
                del model
            except Exception as e:
                logging.warning(f"Error clearing model '{name}': {e}")
        
        # Clear the dictionary
        self.models.clear()
        self.active_model_name = None
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()