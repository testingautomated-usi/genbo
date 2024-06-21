from self_driving.state import State
from test.config import (
    INDIVIDUAL_NAMES,
    STATE_INDIVIDUAL_NAME,
    STATE_PAIR_INDIVIDUAL_NAME,
)
from test.individual import Individual
from test.state_individual import StateIndividual
from test.state_pair_individual import StatePairIndividual


def make_individual(
    individual_name: str,
    start_id: int = 1,
    state: State = None,
    state_other: State = None,
    check_null_state: bool = True,
    mutate_both_members: bool = False,
) -> Individual:

    assert individual_name in INDIVIDUAL_NAMES, "Individual name {} not in {}".format(
        individual_name, INDIVIDUAL_NAMES
    )

    if individual_name == STATE_INDIVIDUAL_NAME:
        if check_null_state:
            assert state is not None, "State cannot be None"
        return StateIndividual(state=state, start_id=start_id)

    if individual_name == STATE_PAIR_INDIVIDUAL_NAME:
        if check_null_state:
            assert state is not None, "State cannot be None"
            assert state_other is not None, "State other cannot be None"
        return StatePairIndividual(
            state1=state,
            state2=state_other,
            start_id=start_id,
            mutate_both_members=mutate_both_members,
        )

    raise RuntimeError("Unknown individual name: {}".format(individual_name))
