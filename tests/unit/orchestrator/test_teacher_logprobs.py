import asyncio
import json

import httpx
import verifiers as vf

from prime_rl.orchestrator import utils as orchestrator_utils
from prime_rl.transport import TrainingSample


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict):
        # Match what AsyncOpenAI exposes — utils.py reads ``str(client.base_url)``.
        self.base_url = "http://fake-host:8000/v1"
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=200,
            content=json.dumps(self._payload).encode(),
            request=request,
        )


def test_compute_teacher_logprobs_uses_inference_generate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [None, {"11": {"logprob": -0.7}}, {"12": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)

        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
        )

        assert result == [[0.0, -0.7, -0.3]]
        assert fake_client.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
                "body": {
                    "model": "teacher-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]

    asyncio.run(_run())


def test_splice_teacher_prompt_replaces_plain_prefix():
    # plain prompt = [plain_prefix ...][blog tail]; cheat prefix is longer.
    plain_prefix = [10, 11, 12]
    cheat_prefix = [10, 11, 99, 98, 12]  # cheatsheet tokens spliced inside the system block
    prompt_ids = plain_prefix + [50, 51, 52]  # blog + gen suffix tail
    out = orchestrator_utils.splice_teacher_prompt(prompt_ids, plain_prefix, cheat_prefix)
    assert out == cheat_prefix + [50, 51, 52]


def test_splice_teacher_prompt_raises_on_prefix_mismatch():
    import pytest

    with pytest.raises(ValueError, match="does not start with the expected plain"):
        orchestrator_utils.splice_teacher_prompt([1, 2, 3, 4], [9, 9, 9], [9, 9, 8, 9])


def test_realign_teacher_logprobs_matches_student_sample_length():
    # teacher scored cheat_prompt(len 5) + completion(len 2): flat length 7.
    teacher_flat = [0.0, -0.1, -0.2, -0.3, -0.4, -1.5, -1.6]
    student_prompt_len = 3
    completion_len = 2
    out = orchestrator_utils.realign_teacher_logprobs(teacher_flat, student_prompt_len, completion_len)
    # 0.0 over the (masked) student prompt positions, teacher completion logprobs after.
    assert out == [0.0, 0.0, 0.0, -1.5, -1.6]
    assert len(out) == student_prompt_len + completion_len  # packer invariant


def test_compute_teacher_logprobs_cheatsheet_splice(monkeypatch):
    async def _run():
        # teacher prompt = cheat_prefix[100,101,102,103] + tail[2]; completion [3,4].
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"101": {"logprob": -0.5}},
                    {"102": {"logprob": -0.6}},
                    {"103": {"logprob": -0.7}},
                    {"2": {"logprob": -0.8}},
                    {"3": {"logprob": -1.1}},
                    {"4": {"logprob": -1.2}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)

        # student plain prompt = plain_prefix[100,101] + tail[2]
        sample = TrainingSample(
            prompt_ids=[100, 101, 2],
            prompt_mask=[True, True, True],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
            cheatsheet_splice=([100, 101], [100, 101, 102, 103]),
        )

        # Teacher scores cheat_prefix + tail + completion.
        assert fake_client.calls[0]["body"]["token_ids"] == [100, 101, 102, 103, 2, 3, 4]
        # Returned logprobs realigned to student sample length (3 prompt + 2 completion).
        assert result == [[0.0, 0.0, 0.0, -1.1, -1.2]]

    asyncio.run(_run())
