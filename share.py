import global_config
from libs.controlnet.cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if global_config.save_memory:
    enable_sliced_attention()
