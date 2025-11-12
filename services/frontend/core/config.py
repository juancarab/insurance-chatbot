from dataclasses import dataclass
from enum import Enum

class Theme(Enum):
    LIGHT = "light"
    DARK = "dark"

@dataclass
class AppConfig:
    api_url: str
    top_k: int
    enable_web_search: bool
    debug: bool
    language: str
    theme: Theme
    auto_scroll: bool
    show_timestamps: bool