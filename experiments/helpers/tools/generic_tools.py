from langchain_core.tools import tool
from datetime import datetime
import pytz  

@tool
def get_current_datetime(timezone_str="America/New_York"):
    """
    Retrieves the current date and time in the specified timezone.

    Args:
        timezone_str (str, optional): A valid IANA timezone string (e.g., "America/New_York", "UTC"). Defaults to "America/New_York".

    Returns:
        datetime: A `datetime` object representing the current date and time in the specified timezone.

    Examples:
        >>> get_current_datetime()
        datetime.datetime(2024, 6, 30, 0, 36, 19, 28, tzinfo=<DstTzInfo 'America/New_York' EDT-1 day, 20:00:00 DST>)
        >>> get_current_datetime("UTC")
        datetime.datetime(2024, 6, 30, 4, 36, 19, 28, tzinfo=<UTC>)
    """
    tz = pytz.timezone(timezone_str) 
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return now