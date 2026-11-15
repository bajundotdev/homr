import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from homr import constants
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.model import MultiStaff, Staff
from homr.point_mapping import PointMapping, chain, identity, undo_crop, undo_resize
from homr.simple_logging import eprint
from homr.staff_dewarping import StaffDewarping, dewarp_staff_image
from homr.staff_parsing_tromr import parse_staff_tromr
from homr.staff_regions import StaffRegions
from homr.system_repair import _layout
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import (
    EncodedSymbol,
    is_lower_position,
    remove_duplicated_symbols,
)
from homr.triplet_restoration import restore_unmarked_triplets
from homr.type_definitions import NDArray

tr_omr_max_height = default_config.max_height
tr_omr_max_width = default_config.max_width


def get_tr_omr_canvas_size(
    image_shape: tuple[int, ...], margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / tr_omr_max_width
    height, width = image_shape[:2]

    # Calculate the new size such that it fits exactly into the
    # tr_omr_max_height and tr_omr_max_width
    # while maintaining the aspect ratio of height and width.

    if height / width > tr_omr_ratio:
        # The height is the limiting factor.
        new_shape = [
            int(width / height * tr_omr_max_height_with_margin),
            tr_omr_max_height_with_margin,
        ]
    else:
        # The width is the limiting factor.
        new_shape = [tr_omr_max_width, int(height / width * tr_omr_max_width)]
    return np.array(new_shape)


def _canvas_y_offset(resized_height: int, margin_top: int = 0, margin_bottom: int = 0) -> int:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    return (tr_omr_max_height_with_margin - resized_height) // 2 + margin_top


def center_image_on_canvas(
    image: NDArray, canvas_size: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

    resized = cv2.resize(image, canvas_size)  # type: ignore

    if is_grayscale:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width),
            255,
            dtype=np.uint8,
        )
    else:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width, 3),
            255,
            dtype=np.uint8,
        )

    x_offset = 0
    y_offset = _canvas_y_offset(resized.shape[0], margin_top, margin_bottom)

    new_image[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ] = resized

    return new_image


def add_image_into_tr_omr_canvas(image: NDArray) -> NDArray:
    new_shape = get_tr_omr_canvas_size(image.shape)
    new_image = center_image_on_canvas(image, new_shape)
    return new_image


def remove_black_contours_at_edges_of_image(gray: NDArray, unit_size: float) -> NDArray:
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold = constants.black_spot_removal_threshold(unit_size)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < threshold or h < threshold:
            continue
        is_at_edge_of_image = x == 0 or y == 0 or x + w == gray.shape[1] or y + h == gray.shape[0]
        if not is_at_edge_of_image:
            continue
        average_gray_intensity = 127
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < average_gray_intensity
        if is_mostly_dark:
            continue
        gray[y : y + h, x : x + w] = 255
    return gray


def _calculate_region(staff: Staff, regions: StaffRegions) -> NDArray:
    x_min = staff.min_x - 2 * staff.average_unit_size
    x_max = staff.max_x + 2 * staff.average_unit_size
    y_min = max(
        staff.min_y - 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_above(staff.min_y),
    )
    y_max = min(
        staff.max_y + 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_below(staff.max_y),
    )
    return np.array([int(x_min), int(y_min), int(x_max), int(y_max)])


def prepare_staff_image(
    debug: Debug, index: int, staff: Staff, staff_image: NDArray, regions: StaffRegions
) -> tuple[NDArray, Staff, PointMapping]:
    """
    Returns the staff image for the transformer, the staff in the coordinates of that image
    and a mapping from the coordinates of that image back to the coordinates of the page.
    """
    region = _calculate_region(staff, regions)
    image_dimensions = get_tr_omr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    page_shape = staff_image.shape
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    undo_scaling = undo_resize(page_shape, staff_image.shape)
    region = np.round(region * scaling_factor)
    eprint("Dewarping staff", index)
    region_step1 = np.array(region) + np.array([-10, -50, 10, 50])
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step1)
    undo_crop1 = undo_crop(*top_left)
    region_step2 = np.array(region) - np.array([*top_left, *top_left])
    top_left = top_left / scaling_factor
    staff = _dewarp_staff(staff, None, top_left, scaling_factor)
    dewarp = dewarp_staff_image(staff_image, staff, index, debug)
    staff_image = dewarp.dewarp(staff_image)
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step2)
    undo_crop2 = undo_crop(*top_left)
    scaling_factor = 1

    eprint("Dewarping staff", index, "done")

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)
    before_canvas_shape = staff_image.shape
    staff_image = center_image_on_canvas(staff_image, image_dimensions)
    canvas_content_shape = (int(image_dimensions[1]), int(image_dimensions[0]))
    to_page = chain(
        undo_crop(0, -_canvas_y_offset(canvas_content_shape[0])),
        undo_resize(before_canvas_shape, canvas_content_shape),
        undo_crop2,
        dewarp.undewarp_point,
        undo_crop1,
        undo_scaling,
    )
    debug.write_image_with_fixed_suffix(f"_staff-{index}_input.jpg", staff_image)
    if debug.debug:
        transformed_staff = _dewarp_staff(staff, dewarp, top_left, scaling_factor)
        transformed_staff_image = staff_image.copy()
        for symbol in transformed_staff.symbols:
            center = symbol.center
            cv2.circle(transformed_staff_image, (int(center[0]), int(center[1])), 5, (0, 0, 255))
            cv2.putText(
                transformed_staff_image,
                type(symbol).__name__,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )
        debug.write_image_with_fixed_suffix(
            f"_staff-{index}_debug_annotated.jpg", transformed_staff_image
        )
    return staff_image, staff, to_page


def _dewarp_staff(
    staff: Staff, dewarp: StaffDewarping | None, region: NDArray, scaling: float
) -> Staff:
    """
    Applies the same transformation on the staff coordinates as we did on the image.
    """

    def transform_coordinates(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x -= region[0]
        y -= region[1]
        if dewarp is not None:
            x, y = dewarp.dewarp_point((x, y))
        x = x * scaling
        y = y * scaling
        return x, y

    return staff.transform_coordinates(transform_coordinates)


def _get_symbol_center(symbol: EncodedSymbol) -> tuple[float, float] | None:
    if symbol.coordinates is None or symbol.rhythm.startswith("chord"):
        return None
    center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    if len(center) < 2 or math.isnan(center[0]) or math.isnan(center[1]):  # noqa: PLR2004
        return None
    return float(center[0]), float(center[1])


def parse_staff_image(
    debug: Debug,
    index: int,
    staff: Staff,
    image: NDArray,
    regions: StaffRegions,
    config: Config,
    page_to_input_image: PointMapping = identity,
) -> list[EncodedSymbol]:
    staff_image, transformed_staff, staff_to_page = prepare_staff_image(
        debug, index, staff, image, regions=regions
    )
    eprint("Running TrOmr inference on staff image", index)
    result = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff, config=config)
    for symbol in result:
        center = _get_symbol_center(symbol)
        if center is not None:
            symbol.image_coordinates = page_to_input_image(staff_to_page(center))
    if debug.debug:
        result_image = staff_image.copy()
        for i, symbol in enumerate(result):
            center = _get_symbol_center(symbol)
            if center is None:
                continue
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(result_image, center_int, 5, color=(0, 0, 255), thickness=2)
            cv2.putText(
                result_image,
                str(i) + ": " + symbol.rhythm,
                (center_int[0], center_int[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )

        debug.write_image_with_fixed_suffix(f"_staff-{index}_output.jpg", result_image)
    return result


@dataclass
class SystemReading:
    """The transcription of one system: is_grandstaff and the symbols of each of its staffs."""

    layout: tuple[bool, ...]
    # Empty if nothing was read in the staff
    staffs: list[list[EncodedSymbol]]


def read_systems(
    debug: Debug,
    staffs: list[MultiStaff],
    image: NDArray,
    config: Config,
    selected_staff: int = -1,
    page_to_input_image: PointMapping = identity,
) -> list[SystemReading]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.

    page_to_input_image maps the coordinates of image back to the image the user provided,
    it's used to fill EncodedSymbol.image_coordinates.
    """
    regions = StaffRegions(staffs)
    i = 0
    result = []
    for staff_index, row in enumerate(staffs):
        symbols_of_row = []
        for staff in row.staffs:
            if selected_staff >= 0 and staff_index != selected_staff:
                eprint("Ignoring staff due to selected_staff argument", i)
                symbols = []
            else:
                symbols = parse_staff_image(
                    debug, i, staff, image, regions, config, page_to_input_image
                )
                if len(symbols) == 0:
                    eprint("Nothing found in staff", i)
            i += 1
            symbols_of_row.append(symbols)
        result.append(SystemReading(_layout(row), symbols_of_row))
    return result


def _common_layout(layouts: Iterable[tuple[bool, ...]]) -> tuple[bool, ...] | None:
    """
    The layout of the systems with the most staffs, they show all parts. None if systems
    of that size differ, then it's unclear which parts there are.
    """
    unique = set(layouts)
    if len(unique) == 0:
        return None
    largest = max(unique, key=len)
    if any(len(layout) == len(largest) and layout != largest for layout in unique):
        return None
    return largest


def _possible_slots(row: tuple[bool, ...], layout: tuple[bool, ...]) -> list[tuple[int, ...]]:
    """The ways in which the staffs of a system can be staffs of the layout, top to bottom."""
    return [
        slots
        for slots in itertools.combinations(range(len(layout)), len(row))
        if all(
            layout[slot] == is_grandstaff for slot, is_grandstaff in zip(slots, row, strict=True)
        )
    ]


def _clefs(symbols: list[EncodedSymbol]) -> tuple[str | None, str | None]:
    """The first clef of the upper and of the lower staff."""
    clefs = [s for s in symbols if s.rhythm.startswith("clef")]
    upper = next((s.rhythm for s in clefs if not is_lower_position(s.position)), None)
    lower = next((s.rhythm for s in clefs if is_lower_position(s.position)), None)
    return upper, lower


def _assign_parts(
    systems: list[SystemReading], layout: tuple[bool, ...]
) -> list[tuple[int, ...]] | None:
    """
    For every system the parts which its staffs belong to, the parts are the staffs of
    the layout. If a system fits in several ways, e.g. one choir staff and the piano
    while the score has two choir staffs, the clefs which the parts have in the other
    systems decide. If they don't, the system gets the lowest parts: in vocal music the
    accompaniment is what is still shown when voices rest. None if a system doesn't fit.
    """
    options = [_possible_slots(system.layout, layout) for system in systems]
    if any(len(slots) == 0 for slots in options):
        return None
    clefs_of_part: list[Counter[tuple[str | None, str | None]]] = [Counter() for _ in layout]
    for system, slots in zip(systems, options, strict=True):
        if len(slots) == 1:
            for slot, symbols in zip(slots[0], system.staffs, strict=True):
                if symbols:
                    clefs_of_part[slot][_clefs(symbols)] += 1

    def clef_matches(system: SystemReading, slots: tuple[int, ...]) -> float:
        matches = 0.0
        for slot, symbols in zip(slots, system.staffs, strict=True):
            seen = clefs_of_part[slot]
            if symbols and seen.total() > 0:
                matches += seen[_clefs(symbols)] / seen.total()
        return matches

    def best(system: SystemReading, slots: list[tuple[int, ...]]) -> tuple[int, ...]:
        return max(slots, key=lambda candidate: (clef_matches(system, candidate), candidate))

    return [best(system, slots) for system, slots in zip(systems, options, strict=True)]


def _multirest_length(symbol: EncodedSymbol) -> int:
    match = re.fullmatch(r"rest_(\d+)m", symbol.rhythm)
    return int(match.group(1)) if match else 1


def _is_measure_content(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith(("note", "rest")) or symbol.rhythm == "measureRest"


def _ends_measure(symbol: EncodedSymbol) -> bool:
    return "barline" in symbol.rhythm or "repeat" in symbol.rhythm


def _count_measures(symbols: list[EncodedSymbol]) -> tuple[int, bool]:
    """The number of measures with notes or rests and if the last one has no bar line yet."""
    count = 0
    in_measure = False
    for symbol in symbols:
        if _is_measure_content(symbol) and not in_measure:
            count += _multirest_length(symbol)
            in_measure = True
        elif _ends_measure(symbol):
            in_measure = False
    return count, in_measure


def _measure_rest(is_grandstaff: bool) -> list[EncodedSymbol]:
    positions = ["upper", "lower"] if is_grandstaff else ["upper"]
    result: list[EncodedSymbol] = []
    for i, position in enumerate(positions):
        if i > 0:
            result.append(EncodedSymbol("chord"))
        result.append(EncodedSymbol("measureRest", position=position))
    return result


def _measure_rests_like(symbols: list[EncodedSymbol], is_grandstaff: bool) -> list[EncodedSymbol]:
    """
    Rests for a staff which is hidden in a system: one per measure of the symbols
    another staff of the system has, with the same barlines.
    """
    result: list[EncodedSymbol] = []
    in_measure = False
    for symbol in symbols:
        if _is_measure_content(symbol) and not in_measure:
            if _multirest_length(symbol) > 1:
                result.append(EncodedSymbol(symbol.rhythm))
            else:
                result.extend(_measure_rest(is_grandstaff))
            in_measure = True
        elif _ends_measure(symbol):
            result.append(EncodedSymbol(symbol.rhythm))
            in_measure = False
    return result


def _pad_measures(
    symbols: list[EncodedSymbol], measures: int, is_grandstaff: bool, ends_open: bool
) -> list[EncodedSymbol]:
    """
    Adds measure rests until the symbols have the given number of measures, so that
    a missed bar line doesn't shift the part against the others in every later system.
    """
    count, in_measure = _count_measures(symbols)
    if count >= measures:
        return symbols
    result = list(symbols)
    for _ in range(measures - count):
        if in_measure:
            result.append(EncodedSymbol("barline"))
        result.extend(_measure_rest(is_grandstaff))
        in_measure = True
    if not ends_open:
        result.append(EncodedSymbol("barline"))
    return result


def _clean_up_voice(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    return restore_unmarked_triplets(remove_duplicated_symbols(symbols))


def _assemble_parts(
    systems: list[SystemReading], slots: list[tuple[int, ...]], layout: tuple[bool, ...]
) -> list[list[EncodedSymbol]]:
    """
    The symbols of every part of the layout. A part which is hidden in a system, or in
    which nothing was found, gets measure rests. All parts get the same number of measures
    in every system.
    """
    parts: list[list[EncodedSymbol]] = [[] for _ in layout]
    for system, system_slots in zip(systems, slots, strict=True):
        shown = {
            slot: symbols
            for slot, symbols in zip(system_slots, system.staffs, strict=True)
            if len(symbols) > 0
        }
        if len(shown) == 0:
            continue
        template = max(shown.values(), key=lambda symbols: _count_measures(symbols)[0])
        measures, ends_open = _count_measures(template)
        for part, is_grandstaff in enumerate(layout):
            symbols = shown.get(part)
            if symbols is None:
                symbols = _measure_rests_like(template, is_grandstaff)
            parts[part].extend(_pad_measures(symbols, measures, is_grandstaff, ends_open))
            parts[part].append(EncodedSymbol("newline"))
    return [_clean_up_voice(part) for part in parts]


def _staffs_on_their_own(systems: list[SystemReading]) -> list[list[EncodedSymbol]]:
    """One voice with every staff after the other."""
    voice: list[EncodedSymbol] = []
    for system in systems:
        for symbols in system.staffs:
            if len(symbols) > 0:
                voice.extend(symbols)
                voice.append(EncodedSymbol("newline"))
    return [_clean_up_voice(voice)]


def parts_of_page(systems: list[SystemReading]) -> list[list[EncodedSymbol]]:
    """
    The parts of a page. The parts are the staffs of the largest system, a system with
    fewer staffs hides some of them.
    """
    if len(systems) == 0:
        return []
    layout = _common_layout(system.layout for system in systems)
    slots = _assign_parts(systems, layout) if layout is not None else None
    if slots is None and len({len(system.layout) for system in systems}) == 1:
        # Same number of staffs, but they differ in grand staffs: staff by staff
        layout = systems[0].layout
        slots = [tuple(range(len(layout)))] * len(systems)
    if layout is None or slots is None:
        eprint("The layout changes on the page, reading every staff on its own")
        return _staffs_on_their_own(systems)
    if any(len(system.layout) < len(layout) for system in systems):
        eprint("Staffs are hidden in some systems, parts of the systems:", slots)
    return _assemble_parts(systems, slots, layout)


def parts_of_score(pages: list[list[SystemReading]]) -> list[list[list[EncodedSymbol]]]:
    """
    The parts of every page, the same parts on every page: a page whose systems all hide
    a staff, e.g. a piano introduction, gets rests for it. Falls back to the parts of each
    page if the systems of the pages don't fit together.
    """
    systems = [system for page in pages for system in page]
    layout = _common_layout(system.layout for system in systems)
    slots = _assign_parts(systems, layout) if layout is not None else None
    if layout is None or slots is None:
        eprint("The pages don't share a layout of their systems, finding the parts per page")
        return [parts_of_page(page) for page in pages]
    eprint("Parts of the score:", len(layout), "parts, staffs of the parts per system:", slots)
    result = []
    start = 0
    for page in pages:
        result.append(_assemble_parts(page, slots[start : start + len(page)], layout))
        start += len(page)
    return result


def parse_staffs(
    debug: Debug,
    staffs: list[MultiStaff],
    image: NDArray,
    config: Config,
    selected_staff: int = -1,
    page_to_input_image: PointMapping = identity,
) -> list[list[EncodedSymbol]]:
    """
    Reads the staffs of a page, see read_systems, and returns the symbols of each part.
    """
    systems = read_systems(debug, staffs, image, config, selected_staff, page_to_input_image)
    return parts_of_page(systems)
