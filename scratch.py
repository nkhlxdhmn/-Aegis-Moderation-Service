import sys

sys.path.append(".")
from backend.pipeline import nsfw

print(nsfw._get_state())
