import logging
import torch

PAD = u'\x00'  # ASCII Null
BOS = u'\x02'  # ASCII Start of Text
EOS = u'\x03'  # ASCII End of Text
FIN = u'\x04'  # ASCII End of Transmission
BSP = u'\x08'  # ASCII Backspace (Separates backward and forward pass of the orthography)
SEP = u'\x1d'  # ASCII Group Seperator (Separates language code from phonemes)

PROFILING = False

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    # torch.xpu is the API for Intel GPU support
    device = torch.device("xpu")
else:
    device = torch.device("CPU")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'Device used: {device}')