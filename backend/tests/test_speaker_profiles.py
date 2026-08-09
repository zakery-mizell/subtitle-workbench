from __future__ import annotations

from unittest import mock

import numpy as np

from backend.app.diarization import SpeakerTurn
from backend.app.speaker_profiles import apply_speaker_profiles, choose_profile_mapping


def test_choose_profile_mapping_finds_one_to_one_voice_matches() -> None:
    similarities = np.asarray(
        [
            [0.88, 0.18, 0.23],
            [0.18, 0.27, 0.83],
            [0.22, 0.89, 0.16],
        ],
        dtype=np.float32,
    )
    matches = choose_profile_mapping(
        similarities,
        profile_indices=[0, 1, 2],
        cluster_labels=["0", "1", "2"],
        names=["Dennis", "Zakery", "Zelalem"],
    )

    assert {(match.name, match.cluster_label) for match in matches} == {
        ("Dennis", "0"),
        ("Zakery", "2"),
        ("Zelalem", "1"),
    }


def test_choose_profile_mapping_rejects_ambiguous_voice() -> None:
    matches = choose_profile_mapping(
        np.asarray([[0.61, 0.57]], dtype=np.float32),
        profile_indices=[0],
        cluster_labels=["0", "1"],
        names=["Guest"],
    )
    assert matches == []


def test_apply_speaker_profiles_relabels_both_turn_lists() -> None:
    exclusive = [
        SpeakerTurn(0.0, 1.0, "0"),
        SpeakerTurn(1.0, 2.0, "1"),
    ]
    raw = list(exclusive)
    profiles = {
        0: ("Alice", np.asarray([1.0, 0.0], dtype=np.float32)),
        1: ("Bob", np.asarray([0.0, 1.0], dtype=np.float32)),
    }

    with (
        mock.patch("backend.app.speaker_profiles._load_requested_profiles", return_value=profiles),
        mock.patch("backend.app.speaker_profiles._decode_mono_16k", return_value=np.ones(64_000, dtype=np.float32)),
        mock.patch(
            "backend.app.speaker_profiles._cluster_reference_audio",
            side_effect=lambda _audio, _turns, label: np.asarray([float(label)], dtype=np.float32),
        ),
        mock.patch(
            "backend.app.speaker_profiles._embedding",
            side_effect=lambda reference: (
                np.asarray([0.0, 1.0], dtype=np.float32)
                if int(reference[0]) == 0
                else np.asarray([1.0, 0.0], dtype=np.float32)
            ),
        ),
    ):
        relabeled, relabeled_raw, matches = apply_speaker_profiles(
            "source.wav", exclusive, raw, ["Alice", "Bob"]
        )

    assert [turn.label for turn in relabeled] == ["1", "0"]
    assert [turn.label for turn in relabeled_raw] == ["1", "0"]
    assert {(match.name, match.cluster_label) for match in matches} == {("Alice", "1"), ("Bob", "0")}
