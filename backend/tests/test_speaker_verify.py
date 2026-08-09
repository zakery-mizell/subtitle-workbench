from backend.app.separation.speaker_verify import choose_verified_speaker


def test_clear_alternative_speaker_wins() -> None:
    chosen = choose_verified_speaker({0: 0.053, 1: 0.423, 2: -0.056}, current_speaker=2)
    assert chosen is not None
    assert chosen[0] == 1
    assert chosen[1] > 0.8


def test_ambiguous_winner_is_left_unchanged() -> None:
    assert choose_verified_speaker({0: 0.30, 1: 0.36, 2: 0.10}, current_speaker=2) is None


def test_current_speaker_winner_is_left_unchanged() -> None:
    assert choose_verified_speaker({0: 0.10, 1: 0.20, 2: 0.48}, current_speaker=2) is None
