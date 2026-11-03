"""
Query understanding: cheap, deterministic entity/season extraction plus an optional
LLM rewrite of conversational follow-ups into standalone search queries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

TEAM_ALIASES = {
    "Chennai Super Kings": ["chennai super kings", "super kings", "chennai", "csk"],
    "Mumbai Indians": ["mumbai indians", "mumbai", "mi"],
    "Royal Challengers Bengaluru": [
        "royal challengers bengaluru", "royal challengers bangalore", "royal challengers",
        "bengaluru", "bangalore", "rcb",
    ],
    "Kolkata Knight Riders": ["kolkata knight riders", "knight riders", "kolkata", "kkr"],
    "Delhi Capitals": ["delhi capitals", "delhi daredevils", "delhi", "dc"],
    "Punjab Kings": ["punjab kings", "kings xi punjab", "kings xi", "punjab", "pbks", "kxip"],
    "Rajasthan Royals": ["rajasthan royals", "rajasthan", "rr"],
    "Sunrisers Hyderabad": ["sunrisers hyderabad", "sunrisers", "hyderabad", "srh"],
    "Gujarat Titans": ["gujarat titans", "gujarat", "titans", "gt"],
    "Lucknow Super Giants": ["lucknow super giants", "super giants", "lucknow", "lsg"],
}
TEAMS = list(TEAM_ALIASES)

# Short abbreviations collide with ordinary words ("mi", "dc", "rr", "gt"), so they only
# match when written in capitals.
_CASE_SENSITIVE = {a for aliases in TEAM_ALIASES.values() for a in aliases if len(a) <= 4}

_PATTERNS: List[Tuple[re.Pattern, str]] = []
for _team, _aliases in TEAM_ALIASES.items():
    for _alias in sorted(_aliases, key=len, reverse=True):
        if _alias in _CASE_SENSITIVE:
            _PATTERNS.append((re.compile(rf"\b{re.escape(_alias.upper())}\b"), _team))
        else:
            _PATTERNS.append((re.compile(rf"\b{re.escape(_alias)}\b", re.IGNORECASE), _team))

_YEAR = re.compile(r"\b(20[0-3]\d)\b")
_CURRENT = re.compile(r"\b(this|current|latest|recent|most recent|ongoing)\s+(season|year|edition|ipl)\b|\bipl\s*19\b", re.I)
_PREVIOUS = re.compile(r"\b(last|previous|prior)\s+(season|year|edition|ipl)\b", re.I)

_IPL_TERMS = re.compile(
    r"\b(ipl|indian premier league|cricket|t20|captain|captained|coach|squad|player|players|"
    r"batter|batsman|bowler|wicket|wickets|runs|innings|century|auction|retain|retained|"
    r"final|playoffs?|qualifier|eliminator|champions?|title|titles|trophy|points table|"
    r"standings|orange cap|purple cap|mvp|stadium|venue|home ground|franchise|owner|"
    r"season|match|matches|won|win|lost|scored|record|highest|lowest|most|kit|sponsor)\b",
    re.I,
)


@dataclass
class QueryAnalysis:
    query: str
    teams: List[str] = field(default_factory=list)
    season: Optional[int] = None
    season_reason: str = ""
    is_ipl: bool = False

    def as_dict(self):
        return {"query": self.query, "teams": self.teams, "season": self.season,
                "season_reason": self.season_reason, "is_ipl": self.is_ipl}


def detect_teams(text: str) -> List[str]:
    found: List[str] = []
    for pattern, team in _PATTERNS:
        if team not in found and pattern.search(text):
            found.append(team)
    return found


def expand_query(query: str) -> str:
    """Append canonical team names for abbreviations/old names ("RCB" -> "Royal Challengers
    Bengaluru"), so BM25 and the embedding model see the vocabulary the chunks use."""
    missing = [t for t in detect_teams(query) if t.lower() not in query.lower()]
    return f"{query} ({', '.join(missing)})" if missing else query


def detect_years(text: str) -> List[int]:
    return sorted({int(y) for y in _YEAR.findall(text) if 2008 <= int(y) <= 2035})


def detect_season(text: str, latest_season: int) -> Tuple[Optional[int], str]:
    years = detect_years(text)
    if len(years) == 1:
        return years[0], f"explicit year {years[0]}"
    if _CURRENT.search(text):
        return latest_season, f"'this/latest season' resolved to {latest_season}"
    if _PREVIOUS.search(text):
        return latest_season - 1, f"'last season' resolved to {latest_season - 1}"
    return None, "no season mentioned" if not years else f"multiple years {years} - no filter"


def analyze(query: str, latest_season: int) -> QueryAnalysis:
    teams = detect_teams(query)
    season, reason = detect_season(query, latest_season)
    is_ipl = bool(teams or season or _IPL_TERMS.search(query))
    return QueryAnalysis(query=query, teams=teams, season=season, season_reason=reason, is_ipl=is_ipl)


class StandaloneQuestion(BaseModel):
    """A self-contained rewrite of the user's latest message."""

    question: str = Field(description="The latest user message rewritten so it can be understood without the conversation. Resolve pronouns like 'they', 'their', 'that season' to explicit team names and years. Keep it short.")


REWRITE_PROMPT = """Rewrite the user's latest message as a standalone question for a search engine about the Indian Premier League (IPL).
Resolve references to earlier turns (teams, players, seasons). If the message is already standalone, return it unchanged.

Conversation:
{history}

Latest message: {question}"""


def rewrite_followup(llm, history: Sequence[Tuple[str, str]], question: str, max_turns: int = 4) -> str:
    """Use a fast LLM with structured output to turn a follow-up into a standalone query."""
    if not history:
        return question
    lines = []
    for role, text in list(history)[-max_turns * 2:]:
        lines.append(f"{role}: {text[:600]}")
    structured = llm.with_structured_output(StandaloneQuestion)
    result = structured.invoke(REWRITE_PROMPT.format(history="\n".join(lines), question=question))
    rewritten = (result.question if result else "").strip()
    return rewritten or question
