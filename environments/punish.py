from .mse import ARESEAMSE


class ARESEAPunish(ARESEAMSE):

    def _reward_fn(self, objective, previous):
        return -objective
