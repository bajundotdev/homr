import fnmatch
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

from homr.main import get_all_image_files_in_folder, main


class TestGetAllImageFilesInFolder(unittest.TestCase):
    def test_case_insensitive_file_system_returns_each_file_once(self) -> None:
        folder = "folder"
        files = [os.path.join(folder, name) for name in ["a.png", "b.PNG", "c.jpg"]]

        def case_insensitive_glob(pattern: str, recursive: bool = False) -> list[str]:
            extension = pattern.rsplit(".", 1)[-1].lower()
            return [f for f in files if fnmatch.fnmatch(f.lower(), f"*.{extension}")]

        with mock.patch("homr.main.glob.glob", side_effect=case_insensitive_glob):
            self.assertEqual(get_all_image_files_in_folder(folder), files)


class TestExitCode(unittest.TestCase):
    def test_failed_image_exits_with_error(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".png") as image,
            mock.patch("sys.argv", ["homr", "--no-title", image.name]),
            mock.patch("homr.main.download_weights"),
            mock.patch("homr.main.process_image", side_effect=Exception("No noteheads found")),
            self.assertRaises(SystemExit) as exit_info,
        ):
            main()
        self.assertEqual(exit_info.exception.code, 1)


class TestRunHomr(unittest.TestCase):
    def test_pages_with_only_a_piano_get_the_parts_of_the_score(self) -> None:
        # Page 1 is a piano introduction, the choir enters on page 2
        from homr.main import PageReading, ProcessingConfig, run_homr  # noqa: PLC0415
        from homr.music_xml_generator import XmlGeneratorArguments  # noqa: PLC0415
        from tests.test_staff_parsing import choir_system, piano_system  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as folder:
            images = [os.path.join(folder, f"page{i}.png") for i in range(2)]
            pages = {
                images[0]: [piano_system(), piano_system()],
                images[1]: [choir_system(), piano_system()],
            }

            def read_page(image: str, config: ProcessingConfig) -> PageReading:
                return PageReading(image.replace(".png", ".musicxml"), "", pages[image])

            with mock.patch("homr.main.read_page", side_effect=read_page):
                run_homr(images, mock.Mock(), XmlGeneratorArguments())

            score = ET.parse(os.path.join(folder, "page0.musicxml")).getroot()  # noqa: S314
        parts = score.findall("part")
        self.assertEqual([8, 8, 8], [len(part.findall("measure")) for part in parts])
        first_measure = [part.findall("measure[1]/note") for part in parts]
        self.assertTrue(all(note.find("rest") is not None for note in first_measure[0]))
        self.assertEqual(["C", "C"], [note.findtext("pitch/step") for note in first_measure[2]])
