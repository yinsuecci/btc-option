from pathlib import Path
from uuid import uuid4

def pytest_configure(config):
    if not config.option.basetemp:
        root=Path(__file__).resolve().parents[1]/'.test-tmp'
        root.mkdir(exist_ok=True)
        config.option.basetemp=str(root/uuid4().hex)
