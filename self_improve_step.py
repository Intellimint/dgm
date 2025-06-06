import argparse
import datetime
import json
import os
import re
import docker

from llm import create_client, get_response_from_llm, extract_json_between_markers
from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe, get_problem_description_prompt
from prompts.diagnose_improvement_prompt import get_diagnose_improvement_prompt
from prompts.testrepo_prompt import get_test_description
from swe_bench.harness import harness
from polyglot.harness import harness as polyglot_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve
from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)

dataset = None
diagnose_model = 'deepseek/deepseek-r1-0528:free'


def parse_test_results(test_output):
    """Parse pytest output and return a summary of failing and errored tests."""
    results = {"failed": [], "error": [], "passed": []}
    pattern = re.compile(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)")
    for line in test_output.splitlines():
        match = pattern.search(line)
        if match:
            name, status = match.groups()
            results[status.lower()].append(name)
    return results


def analyze_test_results(test_output):
    """Analyze pytest output to extract failure messages and priorities."""
    parsed = parse_test_results(test_output)
    failures = parsed["failed"] + parsed["error"]
    lines = test_output.splitlines()
    analysis = []
    error_pattern = re.compile(r"E\s+(.+)")
    for test in failures:
        msg = ""
        try:
            idx = next(i for i, l in enumerate(lines) if test in l)
            for j in range(idx + 1, min(len(lines), idx + 10)):
                m = error_pattern.search(lines[j])
                if m:
                    msg = m.group(1).strip()
                    break
        except StopIteration:
            pass
        analysis.append({"test": test, "error": msg})

    counts = {}
    for item in analysis:
        counts[item["error"]] = counts.get(item["error"], 0) + 1
    for item in analysis:
        item["priority"] = counts[item["error"]]
    analysis.sort(key=lambda x: x["priority"], reverse=True)
    return analysis

def diagnose_problem(entry, commit, root_dir, out_dir, patch_files=[], max_attempts=3, polyglot=False):
    client = create_client(diagnose_model)
    if polyglot:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    else:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        problem_statement = get_problem_description_prompt(response_json, polyglot)
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"Error while diagnosing the problem: {e}")
        if max_attempts > 0:
            return diagnose_problem(
                entry, commit, root_dir, out_dir,
                patch_files=patch_files,
                max_attempts=max_attempts-1,
                polyglot=polyglot,
            )
        else:
            return None
    return problem_statement

def diagnose_improvement(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
        patch_files=[], max_attempts=3,
    ):
    """
    Diagnose the improvement of the model patch.

    Args:
        entry (str): The task entry to improve.
        parent_commit (str): The commit hash of the parent commit.
        root_dir (str): The root directory of the repository.
        model_patch_file (str): The path to the model patch file.
        out_dir (str): The output directory.
        run_id (str): The run id of the self-improvement attempt.
        patch_files (list): The list of patch files before self-improvement.
        max_attempts (int): The maximum number of attempts to diagnose the improvement.
    
    Returns:
        dict: The improvement diagnosis.
    """
    client = create_client(diagnose_model)
    diagnose_sys_message, diagnose_prompt = get_diagnose_improvement_prompt(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id, dataset,
        patch_files=patch_files,
    )
    safe_log(f"Diagnosing the improvement: {diagnose_prompt}")
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        improvement_diagnosis = response_json
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"Error while diagnosing the improvement: {e}")
        if max_attempts > 0:
            return diagnose_improvement(
                entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
                patch_files=patch_files, max_attempts=max_attempts-1,
            )
        else:
            return None
    return improvement_diagnosis

def save_metadata(metadata, output_dir):
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    dnames = harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(5, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=5,
        pred_dname=os.path.join(output_dir, "predictions"),
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start make_report')
    make_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir=output_dir,
        dnames_workers=5,
    )
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(5, len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=5,
            pred_dname=os.path.join(output_dir, "predictions"),
        )
        safe_log('Start make_report more')
        make_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=output_dir,
            dnames_workers=5,
        )
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance'] = overall_performance
        safe_log("End of evaluation more")

def run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    safe_log(f'workers {min(10, len(test_task_list))}')
    dnames = polyglot_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(10, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        pred_dname=os.path.join(output_dir, "predictions"),
        output_dir=output_dir
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = polyglot_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=50,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir
        )
        # metadata['swe_dnames'] = [str(dn) for dn in dnames]
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance_deep'] = overall_performance
        safe_log("End of evaluation more")

def self_improve(
    parent_commit='initial',  # 'initial' if starting from original dgm, else the run_id
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    post_improve_diagnose=True,
    entry=None,
    test_task_list=None,  # None means the entry above only
    # Additional evaluation parameters
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    # Run baseline
    run_baseline=None,
    polyglot=False
):  
    global dataset
    if polyglot:
        with open("polyglot/polyglot_benchmark_metadata.json") as f:
            dataset = json.loads(f.read())
    else:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
        dataset = dataset['test']

    # Variables for this self-improvement attempt
    metadata = {}
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit
    metadata['entry'] = entry
    metadata['test_task_list'] = test_task_list
    metadata['test_more_threshold'] = test_more_threshold
    metadata['test_task_list_more'] = test_task_list_more
    metadata['full_eval_threshold'] = full_eval_threshold
    metadata['run_baseline'] = run_baseline
    metadata['polyglot'] = polyglot

    # Create output directory
    output_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    save_metadata(metadata, output_dir)

    # Initialize Docker client
    try:
        client = docker.from_env()
        safe_log("Docker client initialized successfully")
    except Exception as e:
        safe_log(f"Failed to initialize Docker client: {e}")
        return None

    # Build container
    try:
        container = build_dgm_container(client=client, force_rebuild=force_rebuild)
        if container is None:
            safe_log("Failed to build container")
            return None
        safe_log("Container built successfully")
    except Exception as e:
        safe_log(f"Error building container: {e}")
        return None

    try:
        # Copy files to container
        safe_log("Copying files to container...")
        copy_to_container(container, "coding_agent.py", "/app/coding_agent.py")
        copy_to_container(container, "test_problem.py", "/app/test_problem.py")
        copy_to_container(container, "graph_traversal.py", "/app/graph_traversal.py")
        copy_to_container(container, "tools/", "/app/tools/")
        copy_to_container(container, "utils/", "/app/utils/")
        copy_to_container(container, "prompts/", "/app/prompts/")
        safe_log("Files copied successfully")

        # Run the test problem and capture results
        safe_log("Running test problem...")
        exit_code, output = container.exec_run(
            "python -m pytest test_problem.py -v",
            workdir="/app"
        )
        test_output = output.decode()
        safe_log(f"Test output: {test_output}")

        # Save raw test results
        test_results_file = os.path.join(output_dir, "test_results_iter0.txt")
        with open(test_results_file, "w") as f:
            f.write(test_output)

        parsed_results = parse_test_results(test_output)
        summary_lines = []
        if parsed_results["failed"]:
            summary_lines.append("Failing tests:")
            summary_lines.extend(f"- {name}" for name in parsed_results["failed"])
        if parsed_results["error"]:
            summary_lines.append("Errored tests:")
            summary_lines.extend(f"- {name}" for name in parsed_results["error"])
        test_summary = "\n".join(summary_lines)
        analysis = analyze_test_results(test_output)
        analysis_summary = "\n".join(
            f"{item['test']}: {item['error']} (priority {item['priority']})" for item in analysis
        )

        improvement_tracking = []
        failing_tests = parsed_results["failed"] + parsed_results["error"]
        iteration = 1

        while failing_tests and iteration <= 3:
            # Get the problem description with test results context
            safe_log("Getting problem description...")
            problem_statement = diagnose_problem(
                entry, parent_commit, "/app", output_dir,
                patch_files=[], polyglot=polyglot
            )
            if problem_statement is None:
                safe_log("Failed to get problem statement")
                return None

            enhanced_problem = f"""Problem Statement:
{problem_statement}

Test Summary:
{test_summary}

Failure Analysis:
{analysis_summary}

Full Test Output:
{test_output}

Provide concrete improvements to fix the failing tests above."""

            with open(os.path.join(output_dir, "enhanced_problem_statement.txt"), "w") as f:
                f.write(enhanced_problem)
            copy_to_container(container, os.path.join(output_dir, "enhanced_problem_statement.txt"), "/app/enhanced_problem_statement.txt")

            safe_log(f"Running improvement iteration {iteration}...")
            exit_code, output = container.exec_run(
                f"python coding_agent.py --problem_statement_file enhanced_problem_statement.txt",
                workdir="/app"
            )
            improvement_output = output.decode()
            safe_log(f"Improvement output: {improvement_output}")

            # Run tests again
            exit_code, output = container.exec_run(
                "python -m pytest test_problem.py -v",
                workdir="/app"
            )
            test_output = output.decode()
            safe_log(f"Iteration {iteration} test output: {test_output}")
            with open(os.path.join(output_dir, f"test_results_iter{iteration}.txt"), "w") as f:
                f.write(test_output)

            new_results = parse_test_results(test_output)
            new_failing = new_results["failed"] + new_results["error"]
            fixed = [t for t in failing_tests if t not in new_failing]
            improvement_tracking.append({"iteration": iteration, "fixed_tests": fixed})

            failing_tests = new_failing
            parsed_results = new_results
            summary_lines = []
            if parsed_results["failed"]:
                summary_lines.append("Failing tests:")
                summary_lines.extend(f"- {name}" for name in parsed_results["failed"])
            if parsed_results["error"]:
                summary_lines.append("Errored tests:")
                summary_lines.extend(f"- {name}" for name in parsed_results["error"])
            test_summary = "\n".join(summary_lines)
            analysis = analyze_test_results(test_output)
            analysis_summary = "\n".join(
                f"{item['test']}: {item['error']} (priority {item['priority']})" for item in analysis
            )
            iteration += 1

        metadata['improvement_tracking'] = improvement_tracking


        # Copy results back
        copy_from_container(container, "/app/model_patch.txt", os.path.join(output_dir, "model_patch.txt"))
        copy_from_container(container, "/app/coding_agent.py", os.path.join(output_dir, "coding_agent.py"))

        # Run evaluation
        if polyglot:
            run_harness_polyglot(
                entry, "coding_agent.py",
                [os.path.join(output_dir, "model_patch.txt")],
                num_evals, output_dir, metadata, run_id,
                test_more_threshold, test_task_list, test_task_list_more
            )
        else:
            run_harness_swe(
                entry, "coding_agent.py",
                [os.path.join(output_dir, "model_patch.txt")],
                num_evals, output_dir, metadata, run_id,
                test_more_threshold, test_task_list, test_task_list_more
            )

        # Diagnose improvement
        if post_improve_diagnose:
            improvement_diagnosis = diagnose_improvement(
                entry, parent_commit, "/app",
                os.path.join(output_dir, "model_patch.txt"),
                output_dir, run_id, patch_files=[]
            )
            if improvement_diagnosis:
                metadata['improvement_diagnosis'] = improvement_diagnosis
                save_metadata(metadata, output_dir)

    except Exception as e:
        safe_log(f"Error during self-improvement process: {e}")
        return None
    finally:
        # Cleanup
        try:
            cleanup_container(container)
            remove_existing_container()
        except Exception as e:
            safe_log(f"Error during cleanup: {e}")

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Self-improvement step for the repository.")
    parser.add_argument('--parent_commit', default="initial", type=str, help='Current commit to find the eval results, "initial" if starting from original dgm, else the run_id')
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str, help='Directory to store the output')
    parser.add_argument('--force_rebuild', default=False, action='store_true', help='Force rebuild of the Docker image')
    parser.add_argument('--num_evals', default=1, type=int, help='Repeated number of swe evaluations after self-improvement')
    parser.add_argument('--no_post_improve_diagnose', default=False, action='store_true', help='Skip diagnosing the self-improvement after evaluation')
    parser.add_argument('--entry', default="django__django-10999", type=str, help='Task entry to improve')
    parser.add_argument('--test_task_list', default=None, type=str, help='List of tasks to evaluate the self-improvement')
    parser.add_argument('--polyglot', default=False, action='store_true', help='Run in polyglot mode')
    args = parser.parse_args()

    # Copy cached initial version into experiment dir
    os.system(f"cp -r initial/ {args.output_dir}")

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        post_improve_diagnose=not args.no_post_improve_diagnose,
        entry=args.entry,
        test_task_list=args.test_task_list,
        polyglot=args.polyglot,
    )

if __name__ == "__main__":
    main()
