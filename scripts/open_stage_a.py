"""Launch ONLY the stage-A panel for this independent character."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s1_panel
s1_panel.boot()
