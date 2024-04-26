from ldm.modules.lora import LoRAModule
from ldm.modules.attention import MemoryEfficientCrossAttention, CrossAttention

def inject_lora(model, lora_dim=32):
    print(f'[INFO] inject LoRA (lora_dim={lora_dim})')
    for name, module in model.named_modules():
        if isinstance(module, MemoryEfficientCrossAttention) or isinstance(module, CrossAttention):
            module.to_q = LoRAModule(module.to_q, lora_dim=lora_dim)
            module.to_k = LoRAModule(module.to_k, lora_dim=lora_dim)
            module.to_v = LoRAModule(module.to_v, lora_dim=lora_dim)
            module.to_out[0] = LoRAModule(module.to_out[0], lora_dim=lora_dim)
