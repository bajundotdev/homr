import unittest

from homr.model import MultiStaff, Staff, StaffPoint
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.staff_parsing import (
    SystemReading,
    _assemble_parts,
    _assign_parts,
    _clean_up_voice,
    _count_measures,
    _measure_rests_like,
    parts_of_page,
    parts_of_score,
)
from homr.system_repair import (
    _common_layout,
    _ensure_same_number_of_staffs,
    _unify_grand_staffs,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import read_token_lines


def make_staff(number: float, is_grandstaff: bool = False) -> Staff:
    """A staff 40 high, at 100 * number. Staffs at n and n + 0.5 are close, like a grand staff."""
    y_points = [10 * i + 100 * number for i in range(5)]
    staff = Staff([StaffPoint(0.0, y_points, 0)])
    staff.is_grandstaff = is_grandstaff
    return staff


def make_row(number: float, is_grandstaff: bool = False) -> MultiStaff:
    return MultiStaff([make_staff(number, is_grandstaff)], [])


def make_grandstaff_row(number: int) -> MultiStaff:
    """A grand staff of the staffs number and number + 1."""
    return MultiStaff([make_staff(number).merge(make_staff(number + 1))], [])


def _layouts(result: list[MultiStaff]) -> list[list[bool]]:
    return [[s.is_grandstaff for s in row.staffs] for row in result]


class TestStaffParsing(unittest.TestCase):
    def test_ensure_same_number_of_staffs_already_uniform(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern(self) -> None:
        # A voice and a piano grand staff in every system
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern_grand_staff_first(self) -> None:
        # The piano grand staff above the solo staff in every system
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i, is_grandstaff=True))
            staffs.append(make_row(i + 1))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, False]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_absorbs_inconsistently_pre_merged_row(self) -> None:
        # Only the first system was connected by a bar line
        pre_merged = MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], [])
        staffs = [pre_merged]
        for i in range(2, 8, 2):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_leading_system_with_other_layout(self) -> None:
        staffs = [make_row(0, is_grandstaff=True)]
        staffs += [MultiStaff([make_staff(i), make_staff(i + 1, True)], []) for i in (1, 3, 5)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] + [[False, True]] * 3, _layouts(result))

    def test_ensure_same_number_of_staffs_drops_single_staff_below_last_system(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)] + [make_row(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_splits_pairs_on_page_of_single_staffs(self) -> None:
        staffs = [make_row(0), make_row(1), MultiStaff([make_staff(2), make_staff(3)], [])]
        staffs += [make_row(4), MultiStaff([make_staff(5), make_staff(6)], []), make_row(7)]

        result = _ensure_same_number_of_staffs(staffs)

        # Connected pairs on a page of single staffs are most likely false connections
        self.assertEqual([[False]] * 8, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_certain_systems(self) -> None:
        staffs = [make_row(0), make_row(1), make_row(2)]
        staffs += [MultiStaff([make_staff(3), make_staff(4)], [], certain=True)]
        staffs += [MultiStaff([make_staff(5), make_staff(6)], [], certain=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False]] * 3 + [[False, False]] * 2, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_layout_change(self) -> None:
        # A system where the voice pauses, between systems with voice and piano
        staffs = [
            MultiStaff([make_staff(3 * i), make_staff(3 * i + 1, True)], []) for i in range(4)
        ]
        staffs.insert(2, make_row(12, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 2 + [[True]] + [[False, True]] * 2, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_neighbors_into_common_layout(self) -> None:
        staffs = [
            MultiStaff([make_staff(3 * i), make_staff(3 * i + 1, True)], []) for i in range(3)
        ]
        staffs += [make_row(9), make_row(10, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_single_staffs_with_missed_brace(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in (0, 1, 4, 5)]
        staffs[2:2] = [make_row(2), make_row(2.5)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 5, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_evenly_spaced_single_staffs(self) -> None:
        # Verses for the voice alone after systems of a piano
        staffs = [make_row(i, is_grandstaff=True) for i in range(3)]
        staffs += [make_row(i) for i in range(3, 9)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 3 + [[False]] * 6, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_single_staffs_above_grand_staffs(self) -> None:
        # A violin part whose last systems add a second voice
        staffs = [make_row(i) for i in range(5)]
        staffs += [make_grandstaff_row(i) for i in (5, 7, 9)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False]] * 5 + [[True]] * 3, _layouts(result))

    def test_ensure_same_number_of_staffs_completes_system_with_single_staff(self) -> None:
        # The first grand staff of the second system lost its brace
        system = [make_grandstaff_row(0).staffs[0], make_staff(2), make_staff(3)]
        staffs = [MultiStaff([*system, make_grandstaff_row(4).staffs[0]], [])]
        staffs += [make_row(6)]
        staffs += [MultiStaff([make_staff(7), make_staff(8), make_staff(9)], [])]
        staffs[-1].staffs.append(make_grandstaff_row(10).staffs[0])

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, False, False, True]] * 2, _layouts(result))

    def test_ensure_same_number_of_staffs_splits_wrongly_connected_systems(self) -> None:
        # A crease connected the staffs of several systems
        connected = [make_staff(i, is_grandstaff=i % 2 == 1) for i in range(7)]
        staffs = [MultiStaff(connected, []), make_row(7, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_only_staffs_which_overlap(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]
        right = Staff([StaffPoint(500.0, [10 * i + 1100.0 for i in range(5)], 0)])
        staffs[2:2] = [make_row(10), MultiStaff([right], [])]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 2 + [[False]] * 2 + [[True]] * 2, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_systems_without_common_layout(self) -> None:
        staffs = [
            MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], []),
            MultiStaff([make_staff(2, is_grandstaff=True), make_staff(3)], []),
        ]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual(staffs, result)

    def test_ensure_same_number_of_staffs_finds_grand_staff_found_as_single_staffs(self) -> None:
        # A grand staff was found as two connected single staffs
        staffs = [make_row(0), make_row(1, is_grandstaff=True), make_row(2)]
        staffs += [MultiStaff([make_staff(3), make_staff(3.5)], [])]
        staffs += [make_row(5), make_row(6, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 3, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_system_with_missed_brace(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]
        staffs.append(MultiStaff([make_staff(4), make_staff(5)], []))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 5, _layouts(result))

    def test_ensure_same_number_of_staffs_splits_wrongly_found_grand_staff(self) -> None:
        staffs = [MultiStaff([make_staff(2 * i), make_staff(2 * i + 1)], []) for i in range(3)]
        staffs.append(MultiStaff([make_staff(6).merge(make_staff(7))], []))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, False]] * 4, _layouts(result))

    def test_common_layout_prefers_mix_of_single_and_grand_staffs(self) -> None:
        grand = [False, True, False, True, False, False, False, True]
        staffs = [make_row(i, is_grandstaff) for i, is_grandstaff in enumerate(grand)]

        self.assertEqual((False, True), _common_layout(staffs))

    def test_ensure_same_number_of_staffs_merges_single_staffs_as_close_as_grand_staffs(
        self,
    ) -> None:
        # Two grand staffs whose braces were missed
        staffs = [make_row(0), make_row(0.5), make_row(2), make_row(2.5)]
        staffs += [make_grandstaff_row(4), make_grandstaff_row(6)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_joins_voice_and_piano_of_single_system(self) -> None:
        staffs = [make_row(0), make_row(1, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]], _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_two_grand_staffs_per_system(self) -> None:
        # Piano four hands, the last system has one piano only
        staffs = [MultiStaff([make_staff(i, True), make_staff(i + 1, True)], []) for i in (0, 2)]
        staffs.append(make_row(4, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, True]] * 2 + [[True]], _layouts(result))

    def test_ensure_same_number_of_staffs_merges_only_staffs_as_close_as_grand_staffs(
        self,
    ) -> None:
        staffs = [make_grandstaff_row(0), make_grandstaff_row(2)]
        staffs += [make_row(4), make_row(6)]  # Twice the gap of a grand staff
        staffs += [make_grandstaff_row(8), make_grandstaff_row(10)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 2 + [[False]] * 2 + [[True]] * 2, _layouts(result))

    def test_unify_grand_staffs_keeps_page_if_staffs_dont_overlap(self) -> None:
        right = Staff([StaffPoint(500.0, [10 * i + 600.0 for i in range(5)], 0)])
        staffs = [MultiStaff([make_staff(0), make_staff(1, True)], []) for _ in range(2)]
        staffs.append(MultiStaff([make_staff(4), make_staff(5), right], []))

        self.assertEqual(staffs, _unify_grand_staffs(staffs))

    def test_ensure_same_number_of_staffs_keeps_systems_of_several_grand_staffs(self) -> None:
        staffs = [
            MultiStaff([make_staff(0, True), make_staff(1, True)], []),
            MultiStaff([make_staff(i, True) for i in (2, 3, 4)], []),
        ]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual(_layouts(staffs), _layouts(result))

    def test_unify_grand_staffs_keeps_page_without_most_common_layout(self) -> None:
        staffs = [
            MultiStaff([make_staff(0).merge(make_staff(1))], []),
            MultiStaff([make_staff(2), make_staff(3)], []),
        ]
        self.assertEqual(staffs, _unify_grand_staffs(staffs))

    def test_unify_grand_staffs_keeps_page_if_grand_staff_cant_be_split(self) -> None:
        # Detected as grand staff, but without the lines of two staffs
        staffs = [MultiStaff([make_staff(2 * i), make_staff(2 * i + 1)], []) for i in range(3)]
        staffs.append(MultiStaff([make_staff(6, is_grandstaff=True)], []))
        self.assertEqual(staffs, _unify_grand_staffs(staffs))

    def test_measure_rests_like(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2"),
            EncodedSymbol("note_8", "C4"),
            EncodedSymbol("note_8", "D4"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_4", "E4"),
            EncodedSymbol("repeatEnd"),
        ]

        result = _measure_rests_like(symbols, is_grandstaff=True)

        self.assertEqual(
            ["measureRest", "chord", "measureRest", "barline"]
            + ["measureRest", "chord", "measureRest", "repeatEnd"],
            [s.rhythm for s in result],
        )
        self.assertEqual("lower", result[2].position)

    def test_clean_up_voice_restores_unmarked_triplets(self) -> None:
        # Three eighths in the right hand against each quarter in the left hand
        measure = """note_8 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_8 D5 _ _ _ upper
note_8 E5 _ _ _ upper
barline . . . . ."""
        header = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . ."""
        symbols = read_token_lines([*header.splitlines(), *measure.splitlines() * 3])

        result = _clean_up_voice(symbols)

        right_hand = [s.rhythm for s in result if s.position == "upper" and "note" in s.rhythm]
        self.assertEqual(["note_12"] * 9, right_hand)


# Layouts of systems: is_grandstaff per staff
PIANO = (True,)
CHOIR_AND_PIANO = (False, False, True)


def single_staff(clef: str, measures: int, pitch: str = "C4") -> list[EncodedSymbol]:
    """A staff with a whole note in every measure."""
    symbols = [EncodedSymbol(clef, position="upper"), EncodedSymbol("keySignature_0")]
    symbols.append(EncodedSymbol("timeSignature/4"))
    for _ in range(measures):
        symbols += [EncodedSymbol("note_1", pitch, position="upper"), EncodedSymbol("barline")]
    return symbols


def grand_staff(measures: int) -> list[EncodedSymbol]:
    """A piano grand staff with a whole note in each hand in every measure."""
    symbols = [EncodedSymbol("clef_G2", position="upper"), EncodedSymbol("chord")]
    symbols += [EncodedSymbol("clef_F4", position="lower"), EncodedSymbol("keySignature_0")]
    symbols.append(EncodedSymbol("timeSignature/4"))
    for _ in range(measures):
        symbols += [EncodedSymbol("note_1", "C5", position="upper"), EncodedSymbol("chord")]
        symbols += [EncodedSymbol("note_1", "C3", position="lower"), EncodedSymbol("barline")]
    return symbols


def piano_system(measures: int = 2) -> SystemReading:
    return SystemReading(PIANO, [grand_staff(measures)])


def choir_system(measures: int = 2) -> SystemReading:
    """Soprano and alto, tenor and bass and the piano."""
    staffs = [single_staff("clef_G2", measures, "A4"), single_staff("clef_F4", measures, "A3")]
    return SystemReading(CHOIR_AND_PIANO, [*staffs, grand_staff(measures)])


def measures_per_part(parts: list[list[EncodedSymbol]]) -> list[int]:
    xml = generate_xml(XmlGeneratorArguments(), parts, "")
    return [len(part.findall("measure")) for part in xml.findall("part")]


def pitches(part: list[EncodedSymbol]) -> set[str]:
    return {s.pitch for s in part if s.rhythm.startswith("note")}


class TestScoreParts(unittest.TestCase):
    def test_assign_parts_with_hidden_voice(self) -> None:
        # Voice and piano, in the second system the voice is hidden
        voice_and_piano = SystemReading((False, True), [single_staff("clef_G2", 2), grand_staff(2)])

        slots = _assign_parts([voice_and_piano, piano_system()], (False, True))

        self.assertEqual([(0, 1), (1,)], slots)

    def test_assign_parts_by_clef(self) -> None:
        # One choir staff and the piano, the clef tells which choir staff it is
        tenor_and_bass = SystemReading((False, True), [single_staff("clef_F4", 2), grand_staff(2)])
        soprano_and_alto = SystemReading(
            (False, True), [single_staff("clef_G2", 2), grand_staff(2)]
        )
        systems = [choir_system(), tenor_and_bass, soprano_and_alto]

        slots = _assign_parts(systems, CHOIR_AND_PIANO)

        self.assertEqual([(0, 1, 2), (1, 2), (0, 2)], slots)

    def test_assign_parts_without_clefs_takes_the_lowest_parts(self) -> None:
        # Two voices with the same clef, a system with one of them: the lower one
        two_voices = SystemReading((False, False), [single_staff("clef_G2", 2)] * 2)
        one_voice = SystemReading((False,), [single_staff("clef_G2", 2)])

        self.assertEqual([(0, 1), (1,)], _assign_parts([two_voices, one_voice], (False, False)))

    def test_assign_parts_returns_none_if_a_system_doesnt_fit(self) -> None:
        two_pianos = SystemReading((True, True), [grand_staff(2)] * 2)

        self.assertIsNone(_assign_parts([choir_system(), two_pianos], CHOIR_AND_PIANO))

    def test_parts_of_score_with_staffs_per_system_2_2_4_4_4_2(self) -> None:
        # A piano introduction, the choir enters, a piano interlude; the first page has
        # only the piano
        pages = [
            [piano_system(), piano_system()],
            [choir_system(), choir_system()],
            [choir_system(), piano_system()],
        ]

        result = parts_of_score(pages)

        self.assertEqual([3, 3, 3], [len(parts) for parts in result])
        for parts in result:
            self.assertEqual([4, 4, 4], measures_per_part(parts))
        soprano_and_alto, tenor_and_bass, piano = result[0]
        self.assertEqual(set(), pitches(soprano_and_alto) | pitches(tenor_and_bass))
        self.assertEqual({"C5", "C3"}, pitches(piano))
        self.assertEqual(["A4"] * 4, [s.pitch for s in result[1][0] if s.rhythm == "note_1"])
        self.assertEqual(["A3"] * 2, [s.pitch for s in result[2][1] if s.rhythm == "note_1"])

    def test_parts_of_score_falls_back_to_parts_per_page(self) -> None:
        # The pages don't fit one layout: voice above or below the piano
        voice_above = SystemReading((False, True), [single_staff("clef_G2", 1), grand_staff(1)])
        voice_below = SystemReading((True, False), [grand_staff(1), single_staff("clef_G2", 1)])

        result = parts_of_score([[voice_above], [voice_below]])

        self.assertEqual([2, 2], [len(parts) for parts in result])
        self.assertEqual({"C4"}, pitches(result[0][0]))
        self.assertEqual({"C5", "C3"}, pitches(result[1][0]))

    def test_parts_of_page_with_hidden_choir(self) -> None:
        parts = parts_of_page([piano_system(), piano_system(), choir_system()])

        self.assertEqual([6, 6, 6], measures_per_part(parts))
        self.assertEqual({"A4"}, pitches(parts[0]))

    def test_parts_of_page_reads_staffs_on_their_own_if_layout_is_unclear(self) -> None:
        voice_and_piano = SystemReading((False, True), [single_staff("clef_G2", 1), grand_staff(1)])
        piano_and_voice = SystemReading((True, False), [grand_staff(1), single_staff("clef_G2", 1)])
        voice = SystemReading((False,), [single_staff("clef_G2", 1)])

        parts = parts_of_page([voice_and_piano, piano_and_voice, voice])

        self.assertEqual(1, len(parts))

    def test_staff_without_symbols_gets_rests(self) -> None:
        system = choir_system()
        system.staffs[2] = []

        parts = _assemble_parts([choir_system(), system], [(0, 1, 2)] * 2, CHOIR_AND_PIANO)

        self.assertEqual([4, 4, 4], measures_per_part(parts))
        self.assertEqual(2, len([s for s in parts[2] if s.rhythm == "note_1" and s.pitch == "C3"]))

    def test_parts_get_the_same_number_of_measures(self) -> None:
        # A bar line was missed in the piano, the piano gets a rest at the end
        system = choir_system(measures=3)
        system.staffs[2] = grand_staff(2)

        parts = _assemble_parts([system, choir_system()], [(0, 1, 2)] * 2, CHOIR_AND_PIANO)

        self.assertEqual([5, 5, 5], measures_per_part(parts))
        self.assertEqual((5, False), _count_measures(parts[2]))

    def test_measure_rests_like_keeps_multirests(self) -> None:
        symbols = [EncodedSymbol("rest_4m"), EncodedSymbol("barline")]
        symbols += [EncodedSymbol("note_1", "C4"), EncodedSymbol("barline")]

        result = _measure_rests_like(symbols, is_grandstaff=False)

        self.assertEqual(
            ["rest_4m", "barline", "measureRest", "barline"], [s.rhythm for s in result]
        )
        self.assertEqual((5, False), _count_measures(result))
