from .sequential import ARESEASequential


class ARESEAMSE(ARESEASequential):

    def _objective_fn(self, achieved, desired):
        return ((achieved - desired)**2).mean()
