import itertools

import json

from environments import ARESEASequential


def make_problems(n=100):
    env = ARESEASequential(
        backend="simulation",
        backendargs={"measure_beam": "direct"}
    )

    problems = []
    for _ in range(n):
        env.reset()

        problem = {
            "initial": list(env.backend.actuators.astype("float64")),
            "incoming": {k: float(v) for k, v in env.backend._incoming.parameters.items()},
            "misalignments": list(env.backend.misalignments),
            "desired": list(env.desired.astype("float64"))
        }
        
        problems.append(problem)
    
    return problems


def place_structured_problems(problems):
    point = (0.0, 0.0)
    horizontal = (5e-4, 0.0)
    vertical = (0.0, 5e-4)
    shapes = [point, horizontal, vertical]

    xs = [0.0, -1e-3, 1e-3]
    ys = [0.0, -1e-3, 1e-3]

    for i, (mu_y, mu_x, (sigma_x, sigma_y)) in enumerate(itertools.product(ys, xs, shapes)):
        problems[i]["desired"] = [mu_x, mu_y, sigma_x, sigma_y]
    
    return problems


def main():
    problems = make_problems(n=300)
    problems = place_structured_problems(problems)
    
    with open("problems_3.json", "w") as f:
        json.dump(problems, f, indent=4)


if __name__ == "__main__":
    main()
