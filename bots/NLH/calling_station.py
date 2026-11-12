"""Calling station: always checks or calls. Anchored at Elo 400."""

from pokerarena import Action, BaseBot


class CallingStation(BaseBot):
    """Always checks or calls; never folds or raises."""

    NAME = "station"
    ANCHOR = 400

    def act(self, obs):
        return Action.check() if obs.legal.can_check else Action.call()
