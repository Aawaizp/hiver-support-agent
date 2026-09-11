"""One entry point; command implementations are imported only when needed."""
import runpy
import sys

COMMANDS = {
    "judge-agent": "hiver.judge_agent",
    "verify-results": "hiver.verify_results",
    "eval-baselines-test": "hiver.evaluate_baselines_test",
    "judge-baselines": "hiver.judge_baselines",
    "judge": "hiver.judge",
    "agent": "hiver.agent",
    "retrieve": "hiver.retrieval",
    "baselines": "hiver.baselines",
    "eval-baselines": "hiver.evaluate_baselines",
    "eval-agent": "hiver.evaluate_agent",
    "read-example": "hiver.tools.read_example",
    "test-api": "hiver.tools.test_api",
    "inspect-data": "hiver.tools.inspect_spotify",
    "prepare-data": "hiver.tools.prepare_dataset",
    "prepare-practice": "hiver.tools.prepare_annotation_practice",
    "check-duplicates": "hiver.tools.check_duplicates",
    "clean-retrieval": "hiver.tools.clean_retrieval",
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: py run.py COMMAND [arguments]\n")
        print("Commands:\n  " + "\n  ".join(COMMANDS))
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command: {command}. Use py run.py --help")
    sys.argv = [f"run.py {command}", *sys.argv[2:]]
    runpy.run_module(COMMANDS[command], run_name="__main__")

if __name__ == "__main__":
    main()
