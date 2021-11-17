from .sequential import ARESEASequential


class ARESEAPunish(ARESEASequential):

    def _reward_fn(self, objective, previous):
        return -objective
