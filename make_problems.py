import json
import numpy as np

from environments import ARESEASequential


def make_problems(n=100):
    env = ARESEASequential(
        backend="simulation",
        backendargs={"measure_beam": "direct"}
    )

    problems = []
    for _ in range(n):
        env.next_initial = np.zeros(5)
        env.reset()

        problem = {
            "initial": list(env.backend.actuators),
            "incoming": {k: float(v) for k, v in env.backend._incoming.parameters.items()},
            "misalignments": list(env.backend.misalignments),
            "desired": list(env.desired.astype("float64"))
        }
        
        problems.append(problem)
    
    return problems


def main():
    problems = make_problems(n=300)
    
    with open("problems.json", "w") as f:
        json.dump(problems, f, indent=4)


if __name__ == "__main__":
    main()
