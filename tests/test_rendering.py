import io
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image, ImageDraw

from crowdcanvas import detect_face_bbox, prepare_sprite
from crowd_layout import Figure, CrowdLayout, create_layout, density_map, render_layout, print_dimensions


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.person = Image.new("RGBA", (10, 26), (20, 30, 40, 255))
        self.input = Image.new("RGB", (180, 160), "white")
        ImageDraw.Draw(self.input).rectangle((15, 20, 160, 140), fill="black")

    def test_deterministic_layout_and_separate_export(self):
        opts = dict(density_count=120, scatter_count=0, sprite_height_pct=.035, iterations=2, seed=9)
        first = create_layout(self.input, [self.person], **opts)
        second = create_layout(self.input, [self.person], **opts)
        self.assertEqual(first, second)
        before = first.figures.copy()
        preview = render_layout(first, [self.person], output_size=300, paper_grain=0)
        export = render_layout(first, [self.person], dimensions=(600, 900), paper_grain=0)
        self.assertEqual(first.figures, before)
        self.assertEqual(export.size, (600, 900))
        self.assertEqual(preview.info["figures_placed"], export.info["figures_placed"])

    def test_footprints_and_letter_counter(self):
        ring = Image.new("RGB", (160, 160), "white"); draw = ImageDraw.Draw(ring)
        draw.ellipse((10, 10, 150, 150), fill="black")
        draw.ellipse((45, 45, 115, 115), fill="white")
        gap = .2
        layout = create_layout(ring, [self.person], mode="text", density_count=250,
            scatter_count=0, sprite_height_pct=.035, spacing=gap, iterations=2, seed=2)
        self.assertGreater(len(layout.figures), 30)
        for i, f in enumerate(layout.figures):
            self.assertFalse(45 < f.x * 160 < 115 and 45 < f.y * 160 < 115
                and (f.x * 160 - 80)**2 + (f.y * 160 - 80)**2 < 34**2)
            for g in layout.figures[:i]:
                min_y = (f.height + g.height) * (1 + gap) / 2
                min_x = min_y * self.person.width / self.person.height
                self.assertTrue(abs(f.x - g.x) >= min_x or abs(f.y - g.y) >= min_y)

    def test_empty_and_transparent_inputs(self):
        for image in [Image.new("RGB", (30, 40), "white"), Image.new("RGBA", (30, 40), (255, 255, 255, 0))]:
            layout = create_layout(image, [self.person], density_count=40, scatter_count=0, iterations=1)
            self.assertEqual(len(layout.figures), 0)
        _, density = density_map(Image.new("RGBA", (30, 40), (255, 255, 255, 0)), polarity="light")
        self.assertEqual(density.sum(), 0)

    def test_shadow_away_from_sun_and_attached(self):
        layout = CrowdLayout(1, 1, [Figure(.5, .5, .15, 0, False, (0, 0, 0), 1)], 1, 0, 42)
        base = render_layout(layout, [self.person], output_size=240, shadow_enabled=False, paper_grain=0)
        before = layout.figures.copy()
        for angle, expected_x in [(0, -1), (180, 1)]:
            image = render_layout(layout, [self.person], output_size=240, paper_grain=0,
                sun_angle=angle, shadow_opacity=.5, shadow_softness=0)
            difference = np.abs(np.asarray(base).astype(float) - np.asarray(image).astype(float)).sum(axis=2)
            ys, xs = np.nonzero(difference > 3)
            self.assertGreater(len(xs), 5)
            self.assertGreater((xs.mean() - 120) * expected_x, 0)
            self.assertLess(np.min(np.hypot(xs-120, ys-137)), 12)
        self.assertEqual(layout.figures, before)

    def test_exact_print_size_and_file_dpi(self):
        self.assertEqual(print_dimensions(21, 29.7, 300), (2480, 3508))
        self.assertEqual(print_dimensions(2.54, 5.08, 300), (300, 600))
        with self.assertRaises(ValueError):
            print_dimensions(200, 200, 300)
        image = Image.new("RGB", (300, 600)); data = io.BytesIO()
        image.save(data, "PNG", dpi=(300, 300)); data.seek(0)
        self.assertAlmostEqual(Image.open(data).info["dpi"][0], 300, places=1)

    def test_blank_motif_and_shadow_cleanup_preserve_source(self):
        self.assertIsNone(prepare_sprite(Image.new("RGBA", (100, 100), "white")))
        image = Image.new("RGBA", (100, 100))
        d = ImageDraw.Draw(image);d.rectangle((15, 5, 35, 65), fill=(200, 20, 30, 255))
        d.rectangle((36, 66, 80, 90), fill=(155, 169, 189, 255))
        original = image.tobytes()
        clean = prepare_sprite(image, True)
        self.assertEqual(image.tobytes(), original)
        self.assertLess(clean.height, 70)

    def test_broken_optional_face_detector_does_not_crash(self):
        class BrokenCV:
            class data:
                haarcascades = "/missing/"
            def CascadeClassifier(*args):
                raise SystemError("broken OpenCV")
        with patch.dict("sys.modules", {"cv2": BrokenCV}), warnings.catch_warnings(record=True) as caught:
            self.assertIsNone(detect_face_bbox(self.input))
            self.assertTrue(caught)


if __name__ == "__main__":
    unittest.main()
