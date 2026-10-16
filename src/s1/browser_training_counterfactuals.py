"""Training-owned travel compositions; never imported by browser inference policies."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from dataclasses import dataclass

GOAL_NORMALIZATION = "NFKC-collapse-whitespace-v1"
MAX_COUNTERFACTUAL_EPISODES = 32


def normalized_goal_sha256(goal: str) -> str:
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("A nonempty goal string is required")
    normalized = " ".join(unicodedata.normalize("NFKC", goal).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def validate_excluded_goal_hashes(values) -> frozenset[str]:
    if not isinstance(values, (list, tuple, set, frozenset)) or not values:
        raise ValueError("Counterfactual goal exclusions require nonempty SHA-256 hashes")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("Counterfactual goal exclusions must be lowercase SHA-256 hashes")
    if len(set(values)) != len(values):
        raise ValueError("Counterfactual goal exclusions must be distinct")
    return frozenset(values)


@dataclass(frozen=True)
class TrainingProperty:
    name: str
    city: str
    category: str
    anchor: str
    free_available: bool


# These three identities already belong to the original exposed training page.
TRAINING_PROPERTIES = (
    TrainingProperty("Casa Flora", "Lisbon", "Design", "#casa-flora", True),
    TrainingProperty("The Glasshouse", "Copenhagen", "Design", "#the-glasshouse", True),
    TrainingProperty("Serra Lodge", "Lisbon", "Nature", "#serra-lodge", False),
)


@dataclass(frozen=True)
class CounterfactualTravelEpisode:
    case_id: str
    pair_id: str
    family: int
    start: str
    property: TrainingProperty
    query_required: bool
    free: bool
    goal: str
    # label, upstream kind, optional field text. These are collector actions only.
    initialization: tuple[tuple[str, str, str | None], ...]

    @property
    def expected(self):
        result = {
            "name": self.property.name,
            "hash": self.property.anchor,
            "category": self.property.category,
            "free": self.free,
        }
        if self.query_required:
            result.update(query=self.property.city, searched=True)
        return result


def _goal(property, family, query_required, free, finish):
    category, city, name = property.category, property.city, property.name
    if family == 0:
        cancellation = (
            "with the free-cancellation filter applied"
            if free
            else "without applying the free-cancellation filter"
        )
        request = (
            f"Find {category} stays in {city} using the destination search"
            if query_required
            else f"Choose {category} stays"
        )
        return f"{request}, {cancellation}, then open {name}. {finish}"
    if family == 1:
        cancellation = (
            "keep free cancellation selected" if free else "keep free cancellation unselected"
        )
        request = (
            f"Use destination search to locate {category} accommodation in {city}"
            if query_required
            else f"Set the accommodation category to {category}"
        )
        return f"{request} and {cancellation}. Please show the details of {name}. {finish}"
    if family == 2:
        cancellation = "check Free cancellation" if free else "leave Free cancellation unchecked"
        query = (
            f"Fill Destination with {city} and apply Find stays before opening the result."
            if query_required
            else "Opening the property has no destination-search prerequisite."
        )
        return (
            f"On the stay listing, choose {category} and {cancellation}. "
            f"{query} Open the details for {name}. {finish}"
        )
    cancellation = (
        "using the free-cancellation filter" if free else "leaving the free-cancellation filter off"
    )
    if query_required:
        return (
            f"Look up {category} places around {city} with the destination controls, "
            f"{cancellation}. Bring up {name} when the search is applied. {finish}"
        )
    return (
        f"Bring up {name} from the {category} category, {cancellation}. "
        f"The destination may remain as it is. {finish}"
    )


def make_counterfactual_travel_episodes(*, seed, excluded_goal_hashes):
    """Return all 32 feasible train-only episodes without reading evaluation data.

    Each query/no-query pair has identical guarded initialization. The latter
    leaves query and searched unspecified; it does not demand an empty search.
    Exclusions are rejection-only and cannot supply goals or teacher labels.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("An explicit nonnegative counterfactual seed is required")
    excluded = validate_excluded_goal_hashes(excluded_goal_hashes)
    rng = random.Random(seed)
    # One off/on pair per lexical family. Nature remains represented without
    # inventing the impossible refundable Serra Lodge combination.
    property_indices = ((2, 0), (2, 1), (0, 1), (1, 0))
    episodes = []
    for family, indices in enumerate(property_indices):
        finish = rng.choice(
            ("Stop when its details are visible.", "Finish on its property detail page.")
        )
        for free, property_index in zip((False, True), indices, strict=True):
            property = TRAINING_PROPERTIES[property_index]
            if free and not property.free_available:
                raise ValueError("Requested cancellation profile is infeasible for the catalogue")
            initialize = (("Stay category → " + property.category, "select", None),)
            if free:
                initialize += (("Free cancellation", "click", None),)
            for query_required in (False, True):
                goal = _goal(property, family, query_required, free, finish)
                if normalized_goal_sha256(goal) in excluded:
                    raise ValueError("Counterfactual goal overlaps an exclusion-only goal hash")
                for start in ("listing", "detail"):
                    pair_id = f"counterfactual-{seed}-{family}-{int(free)}-{start}"
                    actions = initialize
                    if start == "detail":
                        actions += (("View " + property.name, "click", None),)
                    episodes.append(
                        CounterfactualTravelEpisode(
                            case_id=f"{pair_id}-query{int(query_required)}",
                            pair_id=pair_id,
                            family=family,
                            start=start,
                            property=property,
                            query_required=query_required,
                            free=free,
                            goal=goal,
                            initialization=actions,
                        )
                    )
    if len(episodes) != MAX_COUNTERFACTUAL_EPISODES:
        raise ValueError("The complete counterfactual schedule must contain exactly 32 episodes")
    rng.shuffle(episodes)
    return tuple(episodes)
