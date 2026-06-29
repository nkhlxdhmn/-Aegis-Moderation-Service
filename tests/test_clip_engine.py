"""Unit tests for SigLIP2 signal aggregation (replaces old OpenCLIP tests)."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from backend.pipeline import clip_engine
from backend.pipeline.clip_engine import ClipAnalysisResult


def _empty_cat_scores() -> dict[str, float]:
    return {cat: 0.0 for cat in clip_engine.CATEGORY_PROMPTS}


def _make_pixel_values(torch_mock: MagicMock) -> MagicMock:
    pv = MagicMock()
    pv.to.return_value = pv
    pv.expand.return_value = pv
    return pv


class ClipEngineTests(TestCase):
    def setUp(self) -> None:
        clip_engine._state = None

    def tearDown(self) -> None:
        clip_engine._state = None

    def test_model_id_constants(self) -> None:
        assert clip_engine.MODEL_ID == "google/siglip2-large-patch16-384"
        assert clip_engine.SIGLIP_MODEL_ID == clip_engine.MODEL_ID

    def test_category_prompts_cover_five_categories(self) -> None:
        self.assertEqual(len(clip_engine.CATEGORY_PROMPTS), 5)

    def test_heritage_prompts_cover_diverse_contexts(self) -> None:
        self.assertGreaterEqual(len(clip_engine.HERITAGE_PROMPTS), 8)

    def test_clip_analysis_result_defaults(self) -> None:
        result = ClipAnalysisResult(
            category_scores={"Religious & Spiritual Heritage": 0.9},
            heritage_score=0.9,
        )
        self.assertEqual(result.safety_scores, {})
        self.assertEqual(result.child_scores, {})
        self.assertEqual(result.promotion_scores, {})

    def test_analyze_content_encodes_image_once_and_reuses_pixel_values(self) -> None:
        """analyze_content should call _encode_image once and reuse pixel_values
        for all sigmoid scoring passes."""
        cat_scores = {cat: 0.5 for cat in clip_engine.CATEGORY_PROMPTS}

        with (
            patch("backend.pipeline.clip_engine._get_state") as get_state,
            patch("backend.pipeline.clip_engine._encode_image") as encode_image,
            patch("backend.pipeline.clip_engine._category_scores_from_pv", return_value=cat_scores),
            patch("backend.pipeline.clip_engine._encode_text_query", return_value=None),
            patch("backend.pipeline.clip_engine._fuse_category_scores", return_value=cat_scores),
            patch("backend.pipeline.clip_engine._sigmoid_scores") as sigmoid_scores,
        ):

            state_mock = MagicMock()
            state_mock.torch.cuda.empty_cache = MagicMock()
            state_mock.heritage_inputs = "heritage"
            state_mock.safety_inputs = "safety"
            state_mock.child_inputs = "child"
            state_mock.promotion_inputs = "promotion"
            get_state.return_value = state_mock

            pv_mock = MagicMock()
            encode_image.return_value = pv_mock

            # _sigmoid_scores is called for heritage, safety, child, promotion
            sigmoid_scores.return_value = {}

            clip_engine.analyze_content("image.jpg", None, "")

        encode_image.assert_called_once_with("image.jpg", state_mock)
        # _sigmoid_scores called 4 times: heritage, safety, child, promotion
        self.assertEqual(sigmoid_scores.call_count, 4)
        # Each call receives the same pixel_values
        for c in sigmoid_scores.call_args_list:
            self.assertIs(c[0][0], pv_mock)

    def test_analyze_content_returns_all_score_groups(self) -> None:
        cat_scores = {cat: 0.5 for cat in clip_engine.CATEGORY_PROMPTS}

        with (
            patch("backend.pipeline.clip_engine._get_state") as get_state,
            patch("backend.pipeline.clip_engine._encode_image"),
            patch("backend.pipeline.clip_engine._category_scores_from_pv", return_value=cat_scores),
            patch("backend.pipeline.clip_engine._encode_text_query", return_value=None),
            patch("backend.pipeline.clip_engine._fuse_category_scores", return_value=cat_scores),
            patch("backend.pipeline.clip_engine._sigmoid_scores") as sigmoid_scores,
        ):

            state_mock = MagicMock()
            state_mock.torch.cuda.empty_cache = MagicMock()
            get_state.return_value = state_mock

            sigmoid_scores.side_effect = [
                {p: 0.7 for p in clip_engine.HERITAGE_PROMPTS},  # heritage pass
                {p: 0.3 for p in clip_engine.SAFETY_PROMPTS},  # safety pass
                {p: 0.1 for p in clip_engine.CHILD_PROMPTS},  # child pass
                {p: 0.2 for p in clip_engine.PROMOTION_PROMPTS},  # promotion pass
            ]

            result = clip_engine.analyze_content("image.jpg", "a caption", "ocr")

        self.assertIsInstance(result, ClipAnalysisResult)
        self.assertEqual(set(result.category_scores), set(clip_engine.CATEGORY_PROMPTS))
        self.assertGreater(result.heritage_score, 0.0)
        self.assertIsInstance(result.safety_scores, dict)
        self.assertIsInstance(result.child_scores, dict)
        self.assertIsInstance(result.promotion_scores, dict)

    def test_ocr_text_and_caption_generate_text_queries(self) -> None:
        """When OCR text and caption are non-empty, _encode_text_query is called twice."""
        cat_scores = {cat: 0.0 for cat in clip_engine.CATEGORY_PROMPTS}

        with (
            patch("backend.pipeline.clip_engine._get_state") as get_state,
            patch("backend.pipeline.clip_engine._encode_image"),
            patch("backend.pipeline.clip_engine._category_scores_from_pv", return_value=cat_scores),
            patch("backend.pipeline.clip_engine._encode_text_query") as encode_text_query,
            patch(
                "backend.pipeline.clip_engine._category_scores_from_text", return_value=cat_scores
            ),
            patch("backend.pipeline.clip_engine._fuse_category_scores", return_value=cat_scores),
            patch("backend.pipeline.clip_engine._sigmoid_scores", return_value={}),
        ):

            state_mock = MagicMock()
            state_mock.torch.cuda.empty_cache = MagicMock()
            get_state.return_value = state_mock
            encode_text_query.return_value = MagicMock()  # non-None â†’ uses text path

            clip_engine.analyze_content("image.jpg", "caption text", "ocr text")

        # Called twice: once for OCR, once for caption
        self.assertEqual(encode_text_query.call_count, 2)

    def test_sigmoid_scores_not_called_when_get_state_fails(self) -> None:
        with (
            patch("backend.pipeline.clip_engine._get_state", side_effect=RuntimeError("no GPU")),
            patch("backend.pipeline.clip_engine._sigmoid_scores") as sigmoid_scores,
        ):
            with self.assertRaises(clip_engine.ModelInferenceError):
                clip_engine.analyze_content("image.jpg", None, "")
        sigmoid_scores.assert_not_called()

    def test_device_is_cuda0(self) -> None:
        self.assertEqual(clip_engine.DEVICE, "cuda:0")

    def test_get_category_scores_delegates_to_analyze_content(self) -> None:
        expected = {cat: 0.42 for cat in clip_engine.CATEGORY_PROMPTS}
        with patch(
            "backend.pipeline.clip_engine.analyze_content",
            return_value=ClipAnalysisResult(
                category_scores=expected,
                heritage_score=0.8,
            ),
        ) as mock_analyze:
            result = clip_engine.get_category_scores("img.jpg", "cap", "ocr")

        mock_analyze.assert_called_once_with("img.jpg", "cap", "ocr")
        self.assertEqual(result, expected)
