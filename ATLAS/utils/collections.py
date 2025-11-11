"""Collection helpers shared across services and UI components."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Sequence

_StringTypes = (str, bytes, bytearray)


def normalize_sequence(
    value: Any,
    *,
    as_tuple: bool = False,
    copy_items: bool = False,
    allow_strings: bool = False,
    accept_scalar: bool = False,
    coerce_mapping_values: bool = True,
    transform: Callable[[Any], Any] | None = None,
    filter_none: bool = True,
    filter_falsy: bool = False,
) -> Sequence[Any]:
    """Return an iterable of normalised items extracted from *value*.

    Parameters
    ----------
    value:
        Candidate value to normalise into a flat sequence.
    as_tuple:
        When ``True`` the function returns a :class:`tuple`, otherwise a
        :class:`list` is produced.
    copy_items:
        When ``True`` items are deep-copied before being appended to the
        result which prevents callers from mutating shared references.
    allow_strings:
        Controls whether string-like values should be treated as iterables.
        When ``False`` they are considered scalars instead of sequences.
    accept_scalar:
        Controls whether non-iterable values (or string-like values when
        ``allow_strings`` is ``False``) should be coerced into a single-item
        collection.  When ``False`` scalars are ignored.
    coerce_mapping_values:
        When ``True`` mapping objects yield their ``values()`` during
        normalisation.  When ``False`` mappings are treated like any other
        iterable.
    transform:
        Optional callable applied to each value before it is added to the
        result.
    filter_none:
        When ``True`` ``None`` values are skipped both before and after the
        optional transformation.
    filter_falsy:
        When ``True`` values evaluating to ``False`` (after the optional
        transformation) are skipped.
    """

    if value is None:
        return tuple() if as_tuple else []

    if coerce_mapping_values and isinstance(value, Mapping):
        iterable: Any = value.values()
    else:
        iterable = value

    is_iterable = isinstance(iterable, Iterable)
    is_string_like = isinstance(iterable, _StringTypes)

    if is_iterable and (allow_strings or not is_string_like):
        iterator = iterable
    elif accept_scalar:
        iterator = [iterable]
    else:
        return tuple() if as_tuple else []

    result: list[Any] = []

    for item in iterator:
        if filter_none and item is None:
            continue

        transformed = transform(item) if transform is not None else item

        if filter_none and transformed is None:
            continue
        if filter_falsy and not transformed:
            continue

        result.append(copy.deepcopy(transformed) if copy_items else transformed)

    return tuple(result) if as_tuple else result


__all__ = ["normalize_sequence"]
