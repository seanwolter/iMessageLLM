"""
Output formatting utilities for enhanced terminal display.

This module provides consistent styling, colors, progress bars, and formatting
functions for creating professional-looking terminal output.
"""

import time
from typing import Optional, Union


# Constants
DEFAULT_HEADER_WIDTH = 80
DEFAULT_SECTION_WIDTH = 60
DEFAULT_PROGRESS_BAR_LENGTH = 40
DEFAULT_INDENT = 2


class Colors:
    """ANSI color codes for terminal output."""
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Standard colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


# Header and Section Functions
def print_header(title: str, char: str = '=', width: int = DEFAULT_HEADER_WIDTH) -> None:
    """
    Print a formatted header with consistent styling.
    
    Args:
        title: The header title to display
        char: Character to use for the border (default: '=')
        width: Width of the header in characters (default: 80)
    """
    if not title.strip():
        raise ValueError("Title cannot be empty or whitespace only")
    
    border = char * width
    print(f"\n{Colors.CYAN}{Colors.BOLD}{border}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{border}{Colors.RESET}")


def print_section(title: str, width: int = DEFAULT_SECTION_WIDTH) -> None:
    """
    Print a section header.
    
    Args:
        title: The section title to display
        width: Width of the section header in characters (default: 60)
    """
    if not title.strip():
        raise ValueError("Title cannot be empty or whitespace only")
    
    print(f"\n{Colors.YELLOW}{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(f"{Colors.YELLOW}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─' * width}{Colors.RESET}")


# Information Display Functions
def print_info(label: str, value: Union[str, int, float], unit: str = '', indent: int = DEFAULT_INDENT) -> None:
    """
    Print formatted information with consistent styling.
    
    Args:
        label: The label for the information
        value: The value to display (will be formatted if numeric)
        unit: Optional unit to append to the value
        indent: Number of spaces to indent (default: 2)
    """
    spaces = ' ' * indent
    
    # Format numbers with commas, but leave strings as-is
    if isinstance(value, (int, float)):
        formatted_value = f"{value:,}"
    else:
        formatted_value = str(value)
    
    print(f"{spaces}{Colors.WHITE}{label}:{Colors.RESET} {Colors.GREEN}{formatted_value}{Colors.RESET}{Colors.DIM}{unit}{Colors.RESET}")


def print_progress(
    current: int, 
    total: int, 
    label: str = 'Progress', 
    extra_info: str = '', 
    start_time: Optional[float] = None
) -> None:
    """
    Print enhanced progress bar with timing information.
    
    Args:
        current: Current progress value
        total: Total expected value
        label: Label for the progress bar (default: 'Progress')
        extra_info: Additional information to display
        start_time: Start time for ETA calculation (Unix timestamp)
    """
    if total <= 0:
        raise ValueError("Total must be greater than 0")
    if current < 0:
        raise ValueError("Current must be non-negative")
    
    # Clamp current to not exceed total
    current = min(current, total)
    
    percentage = (current / total) * 100
    filled_length = int(DEFAULT_PROGRESS_BAR_LENGTH * current // total)
    bar = '█' * filled_length + '░' * (DEFAULT_PROGRESS_BAR_LENGTH - filled_length)
    
    timing_info = ''
    if start_time and current > 0:
        elapsed = time.time() - start_time
        eta = (elapsed / current) * (total - current)
        timing_info = f" | ETA: {format_duration(eta)} | Elapsed: {format_duration(elapsed)}"
    
    print(
        f"\r{Colors.BLUE}{label}:{Colors.RESET} "
        f"|{Colors.GREEN}{bar}{Colors.RESET}| "
        f"{Colors.YELLOW}{percentage:6.1f}%{Colors.RESET} "
        f"({current:,}/{total:,}){extra_info}{timing_info}",
        end='', flush=True
    )


# Message Functions
def print_success(message: str) -> None:
    """
    Print success message with green styling.
    
    Args:
        message: The success message to display
    """
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """
    Print error message with red styling.
    
    Args:
        message: The error message to display
    """
    print(f"{Colors.RED}{Colors.BOLD}✗ Error: {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """
    Print warning message with yellow styling.
    
    Args:
        message: The warning message to display
    """
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Warning: {message}{Colors.RESET}")


# Formatting Utility Functions
def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1.5m", "2.3h")
    """
    if seconds < 0:
        return "0.0s"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_file_size(bytes_size: Union[int, float]) -> str:
    """
    Format file size in human readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB", "2.3 GB")
    """
    if bytes_size < 0:
        return "0.0 B"
    
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
