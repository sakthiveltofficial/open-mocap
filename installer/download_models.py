# Thin entry point the add-on calls before a Pro capture on the ONNX
# Runtime path (--ensure-backbone-onnx). The logic lives in install.py.
import sys
from install import main

if __name__ == "__main__":
    sys.exit(main())
