import json

from environments import ARESEAOneStep, ARESEASequential


def make_problems(n=100):
    env = ARESEASequential(
        backend="simulation",
        initial="reset",
        backendargs={
            "incoming": "random",
            "measure_beam": "direct",
            "misalignments": "none"
        }
    )

    problems = []
    for _ in range(n):
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
    problems = make_problems(n=100)
    
    with open("problems.json", "w") as f:
        json.dump(problems, f, indent=4)


if __name__ == "__main__":
    main()
