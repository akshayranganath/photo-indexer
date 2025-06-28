"""
photo_indexer.utils.geo
~~~~~~~~~~~~~~~~~~~~~~~

Tiny wrapper around **geopy** that turns GPS coordinates into a short
human-readable place string (“Paris, France”, “Yosemite National Park”,
…).

Why roll our own wrapper?

* The pipeline must stay *offline-friendly*; we therefore
  1. try an in-memory **LRU cache** first,
  2. attempt a single call to OpenStreetMap-Nominatim,
  3. fall back to *None* if no network or the service rate-limits.

Public API
----------

``reverse_geocode(lat: float, lon: float, /, language="en") -> str | None``

The function is deliberately synchronous; use a thread pool if you need
concurrency.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderServiceError
except ModuleNotFoundError:  # geopy is an optional dependency
    Nominatim = None  # type: ignore

__all__ = ["reverse_geocode"]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
# initialise the OSM geocoder                                                 #
# ---------------------------------------------------------------------------#
if Nominatim:
    _GEOCODER = Nominatim(user_agent="photo-indexer/0.1", timeout=5)
else:  # pragma: no cover
    _GEOCODER = None


# ---------------------------------------------------------------------------#
# public helper                                                               #
# ---------------------------------------------------------------------------#
@lru_cache(maxsize=4_096)
def reverse_geocode(lat: float, lon: float, language: str = "en") -> Optional[str]:
    """
    Convert *lat, lon* to a concise `"City, Country"` string.

    * **LRU-cached** – repeated calls for the same coordinates are free.
    * Returns *None* on any network error or if *geopy* is unavailable.
    * Nominatim usage policy allows 1 req / second.  The pipeline calls
      this at most once per image; worker threads share the global cache.
    """
    if _GEOCODER is None:
        _log.debug("geopy not installed; cannot reverse-geocode.")
        return None

    try:
        loc = _GEOCODER.reverse(
            (lat, lon),
            language=language,
            exactly_one=True,
            addressdetails=False,
            zoom=10,  # ~city / park resolution
        )
    except (GeocoderServiceError, Exception) as exc:  # pylint: disable=broad-except
        _log.debug("Reverse-geocode failed: %s", exc)
        return None

    if loc is None:
        return None

    # Extract a compact “town, country” or “name, country” string
    if hasattr(loc, "raw") and isinstance(loc.raw, dict):
        addr = loc.raw.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("village") or ""
        state = addr.get("state", "")
        country = addr.get("country", "")
        tokens = [city or state, country]
        return ", ".join(filter(None, tokens)) or loc.address

    return loc.address
