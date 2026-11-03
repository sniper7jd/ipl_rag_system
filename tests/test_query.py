from src.retrieval.query import analyze, detect_season, detect_teams, expand_query


def test_aliases_and_former_names():
    assert detect_teams("How did RCB do?") == ["Royal Challengers Bengaluru"]
    assert detect_teams("Royal Challengers Bangalore in 2016") == ["Royal Challengers Bengaluru"]
    assert detect_teams("Kings XI Punjab history") == ["Punjab Kings"]
    assert set(detect_teams("CSK vs MI rivalry")) == {"Chennai Super Kings", "Mumbai Indians"}


def test_short_aliases_are_case_sensitive():
    assert detect_teams("give me a summary in a mi-nute") == []
    assert detect_teams("the gt of the season") == []
    assert detect_teams("GT won") == ["Gujarat Titans"]


def test_season_resolution():
    assert detect_season("Who won IPL 2026?", 2026)[0] == 2026
    assert detect_season("top scorer this season", 2026)[0] == 2026
    assert detect_season("who won last season", 2026)[0] == 2025
    assert detect_season("compare 2008 and 2026", 2026)[0] is None
    assert detect_season("how many titles has CSK won", 2026)[0] is None


def test_router_signal():
    assert analyze("hi there!", 2026).is_ipl is False
    assert analyze("thanks a lot", 2026).is_ipl is False
    assert analyze("Who captained KKR?", 2026).is_ipl is True


def test_expand_query_adds_canonical_names():
    assert expand_query("RCB top scorer") == "RCB top scorer (Royal Challengers Bengaluru)"
    assert expand_query("Mumbai Indians squad") == "Mumbai Indians squad"
