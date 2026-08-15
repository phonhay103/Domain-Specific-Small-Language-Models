# ==========================================
# Extracted from CH07_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Evaluation Python Code Generation with CodeGen 350M mono Using the ReCode Approach
# This notebook is a companion of chapter 7 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to show an approach to evaluate the quality of the Python code generated through a [CodeGen 350 M mono](https://huggingface.co/Salesforce/codegen-350M-mono) model. The code in this notebook has been derived and adapted from the ReCode paper. While the CodeGen 350M mono model is evaluated here, the exact same approach applies to other models for Python code generation available in the Hugging Face Hub. Python is the only programming language that could be evaluate using the code in this notebook. Execution of the code cells of this notebooks requires hardware acceleration (GPU).  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# The correctness evaluation steps depend on the OpenAI's HumanEval package. It isn't available through any Python package manager, so it needs to be installed from source.
# ------------------------------------------------------------

# !git clone https://github.com/openai/human-eval
# %cd human-eval
# !pip install .
# %cd ..

# ------------------------------------------------------------
# Download the preprocessed HumanEval dataset from the ReCode GitHub repo.
# ------------------------------------------------------------

# !mkdir -p datasets/nominal/
# %cd ./datasets/nominal/
# !wget https://raw.githubusercontent.com/amazon-science/recode/refs/heads/main/datasets/nominal/HumanEval.jsonl
# %cd ../..

# ------------------------------------------------------------
# Load the CodeGen 350 M mono model and tokenizer from the HF's Hub. The model weights are then loaded into GPU memory.
# ------------------------------------------------------------

from transformers import AutoTokenizer

model_id = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_id)

import torch
from transformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# ------------------------------------------------------------
# Setup the configuration to be used for the model at code generation time.
# ------------------------------------------------------------

from transformers import GenerationConfig

generation_config = GenerationConfig(
    pad_token_id=50256,
    truncation=True,
    max_length=1000,
    max_context_length=1000,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True
)

# ------------------------------------------------------------
# Define two functions to load the prompts from the JSON file downloaded from the ReCode repo.
# ------------------------------------------------------------

import json
import gzip

def read_problems(eval_file):
    return {str(task["task_id"]): task for task in stream_jsonl(eval_file)}

def stream_jsonl(filename):
      with open(filename, "r") as fp:
          for line in fp:
              if any(not x.isspace() for x in line):
                  yield json.loads(line)

# ------------------------------------------------------------
# Load the prompt list from the JSON file and then create a single task file for each one.
# ------------------------------------------------------------

problems_file_path = 'datasets/nominal/HumanEval.jsonl'
problems = read_problems(problems_file_path)
num_samples = 1
batch_size = 1

import os

output_dir = 'output_dir'
fpath_format = os.path.join(
        output_dir, "output", "taskid-{task_idx}-gen{completion_idx}.json"
    )

# ------------------------------------------------------------
# Define a function to count the non-empty prompt samples.
# ------------------------------------------------------------

def count_files_present_nonemtpy(list_fname):
    count = 0
    for fname in list_fname:
        if os.path.isfile(fname):
            f = open(fname, "r", encoding="utf8")
            s = f.read()
            f.close()
            if not s == "":
                count += 1
    return count, len(list_fname)

# ------------------------------------------------------------
# Define a function to get token position from a string.
# ------------------------------------------------------------

def get_token_position_by_string(target_str, outputs, tokenizer, skip_special_tokens):
    for position in range(1, len(outputs) + 1):
        gen_str = tokenizer.decode(
            outputs[:position],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        if gen_str.rstrip() == target_str.rstrip():
            return position  # not including outputs[position]
        if gen_str.startswith(target_str) and target_str != "":
            print("Cannot find an exact match, use approx!")
            print(f"output length: {len(outputs)}")
            print(target_str)
            print("-----------------------")
            print(gen_str)
            return position
    if target_str.rstrip() == "":
        if target_str == "":
            print("generated empty string!")
        else:
            print("generated only white space!")
        return 0
    print(f"output length: {len(outputs)}")
    print(target_str)
    print("-----------------------")
    print(gen_str)
    raise RuntimeError("Cannot match prefix returned by AST.")

# ------------------------------------------------------------
# Define a function to assess if the generated code is a valid Python code. It uses the Python native *ast* package.
# ------------------------------------------------------------

import ast

def is_valid_python(code):
    try:
        try:
            pared_code = ast.parse(code)
        except SyntaxError:
            return False
    except Exception as e:
        print("Exception: ", e)
        return False
    return pared_code

# ------------------------------------------------------------
# Define a function that grab entire function code from ast.
# ------------------------------------------------------------

def get_function_from_ast(parsed_code, code, option="func_ast_last"):
    assert option in [
        "func_ast_first",
        "func_ast_last",
    ], f"Invalid post process option {option}"
    for i in range(len(parsed_code.body)):
        idx = -i - 1 if option == "func_ast_last" else i
        if type(parsed_code.body[idx]) == ast.FunctionDef:
            break
        idx = None
    assert idx is not None, "No function found"
    function_segment = ast.get_source_segment(code, parsed_code.body[idx])
    position = code.find(function_segment)
    function_segment_plus_previous = code[: position + len(function_segment)]
    return function_segment_plus_previous

# ------------------------------------------------------------
# Define the function that handles how to decode the generated code and checks that it is valid Python code.
# ------------------------------------------------------------

def filter_valid_code(
    true_str_input,
    execution_prompt,
    inputs,
    sequences,
    initial_context_length,
    tokenizer,
    task_id=None,
    has_special_tokens=False,
    post_process="greedy",
    replace_unk=False,
    skip_special_tokens=True,
    mean_logp=None,
    use_language_tag=0,
):
    """
    Due to tokenizer non lossless-ness, the decoded original prompt and
    the real original prompt are not the same.

    Due to constrained generation, input tokens not not necessarily match
    with the new input tokens (but match by characters instead)
    """
    samples = []
    # need both to handle CG / non losslessness of tokenizer
    decoded_context_string = tokenizer.batch_decode(
        inputs[:, use_language_tag:initial_context_length],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]
    decoded_original_prompt = tokenizer.batch_decode(
        inputs[:, use_language_tag:],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]
    processed_prompt = decoded_context_string

    assert execution_prompt is None, "only support execution_prompt is None here"
    processed_execution_prompt = processed_prompt

    output_lists = sequences[:, initial_context_length:]
    for sample_id, outputs in enumerate(output_lists):
        is_valid = False
        for position in range(len(outputs), 0, -1):
            gen_up_to_pos_toks = outputs[:position]
            gen_up_to_pos_str = tokenizer.decode(
                gen_up_to_pos_toks,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            origin_pred = gen_up_to_pos_str
            code = (
                processed_execution_prompt + gen_up_to_pos_str
            )  # something is off for python
            parsed_code = is_valid_python(code)
            if parsed_code:
                is_valid = True
                # print(f"valid at position {position} / {len(outputs) - 1}. ")
                if post_process != "greedy":
                    try:
                        function_segment_plus_previous = get_function_from_ast(
                            parsed_code,
                            code,
                            option=post_process,
                        )
                        generated_part = function_segment_plus_previous[
                            len(processed_execution_prompt) :
                        ]
                    except Exception as e:
                        print("Something went wrong...", e)
                        generated_part = gen_up_to_pos_str
                elif post_process == "greedy":
                    generated_part = gen_up_to_pos_str
                else:
                    assert False, f"post processing method {post_process} not supported"

                if task_id is None:
                    return generated_part
                if mean_logp is None:
                    score = None
                else:
                    if post_process != "greedy":
                        position = get_token_position_by_string(
                            generated_part,
                            outputs,
                            tokenizer,
                            skip_special_tokens,
                        )
                    if position == 0:
                        score = -1e8
                    else:
                        score = mean_logp[sample_id][position - 1]

                samples.append(
                    dict(
                        task_id=task_id,
                        completion=(processed_prompt + generated_part)[
                            len(decoded_original_prompt) :
                        ],
                        ori_pred=(processed_prompt + origin_pred)[
                            len(decoded_original_prompt) :
                        ],
                        input=true_str_input,
                        mean_logp=score,
                    )
                )
                break
        if not is_valid:
            predictions = tokenizer.decode(
                outputs,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            origin_pred = predictions
            print("Warning - no valid substring")
            if task_id is None:
                return predictions
            samples.append(
                dict(
                    task_id=task_id,
                    completion=(processed_prompt + predictions)[
                        len(decoded_original_prompt) :
                    ],
                    ori_pred=(processed_prompt + origin_pred)[
                        len(decoded_original_prompt) :
                    ],
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )

    return samples

# ------------------------------------------------------------
# Create the output directory. It would be used to save the generated code and the evaluation results too.
# ------------------------------------------------------------

# !mkdir -p output_dir/output

# ------------------------------------------------------------
# Start the code generation process. Each sample prompt in the HumanEval dataset is perturebed before being tokenized and sent to the model for Python code generation. At each step, the generated code is assessed to vefify that it is really valid code.
# ------------------------------------------------------------

from tqdm import tqdm

override_previous_results = False
for enum_idx, task_id in enumerate(tqdm(problems)):
        # assume TaskName/ID format. The id part needs not be integer.
        task_idx = task_id.split("/")[1]
        if not override_previous_results:
            fnames = [
                fpath_format.format(task_idx=task_idx, completion_idx=_idx)
                for _idx in range(num_samples)
            ]
            count, all_count = count_files_present_nonemtpy(fnames)
            if count == all_count:
                print(
                    f"Result caching mode: Skipping case {task_id}. Generated all {all_count}"
                )
                continue
            else:
                print(
                    f"Result caching mode: Only {count} out of {all_count} were generated. Regenerating task {task_id}"
                )
        execution_prompt = None
        prompt = problems[task_id]["prompt"]
        completion_idx = -1
        for i in range(0, num_samples, batch_size):
          num_return_sequences = min(num_samples - i, batch_size)
          generation_config = GenerationConfig(
              pad_token_id=50256,
              truncation=True,
              max_length=1000,
              max_context_length=1000,
              use_cache=True,
              return_dict_in_generate=True,
              output_scores=True
          )
          inputs = tokenizer(prompt, return_tensors="pt").to(device)
          input_ids = inputs.input_ids
          output_dict = model.generate(input_ids,
                                        generation_config=generation_config)

          sequences = output_dict.sequences
          initial_context_length = len(sequences[0]) - len(output_dict.scores)

          predictions_post_eos = filter_valid_code(
                        true_str_input=prompt,
                        execution_prompt=execution_prompt,
                        inputs=input_ids,
                        sequences=sequences,
                        initial_context_length=initial_context_length,
                        tokenizer=tokenizer,
                        task_id=task_id,
                        post_process='func_ast_first',
                        skip_special_tokens=True,
                        mean_logp=None,
                    )


          for prediction in predictions_post_eos:
              completion_idx += 1
              fpath = fpath_format.format(
                  task_idx=task_idx, completion_idx=completion_idx
              )
              prediction["language"] = 'python'
              with open(fpath, "w", encoding="utf8") as _f:
                  json.dump(prediction, _f)

# ------------------------------------------------------------
# ### Correctness Evaluation Process
# ------------------------------------------------------------

# ------------------------------------------------------------
# The code in the cell below is to overwrite the same from the HumanEval package, as the line responsible for the execution of the generated code is commented in the original for safety reasons. Please be careful when evaluating Python code generated by LLMs.
# ------------------------------------------------------------

from typing import Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile


def custom_check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                problem["prompt"] + completion + "\n" +
                problem["test"] + "\n" +
                f"check({problem['entry_point']})"
            )

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                      exec(check_program, exec_globals)
                      result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None

# ------------------------------------------------------------
# The code in the cell below is to implement a custom function to run the code correctness steps, to overcome the limitation in the HumanEval library explained above.
# ------------------------------------------------------------

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools

import numpy as np
import tqdm

from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def custom_evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(custom_check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k

# ------------------------------------------------------------
# Execute the code correctness evaluation on the 164 samples. The results are aggregated and saved in a single JSON file. The *passed* attribute contains the result of a test (true or false are the only possible values for it).
# ------------------------------------------------------------

import fire
import sys
import glob
import os

def entry_point(
    sample_dir: str = '/content/output_dir/output/',
    k: str = "1,10,100",
    n_workers: int = 2,
    timeout: float = 3.0,
    problem_file: str = '/content/datasets/nominal/HumanEval.jsonl',
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))

    # Create a list of all generated sample files
    sample_files = glob.glob(os.path.join(sample_dir, '*.json'))

    # Read samples from all generated files and combine them into a list
    samples = []
    for sample_file in sample_files:
        with open(sample_file, "r", encoding="utf8") as f:
            try:
                samples.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {sample_file}")
                continue

    # Create a dummy samples.jsonl file for the evaluation function
    # In a real scenario, you might want to process the samples directly
    # without writing to an intermediate file.
    temp_samples_file = '/content/output_dir/temp_samples.jsonl'
    with open(temp_samples_file, 'w', encoding='utf8') as f:
        for sample in samples:
            json.dump(sample, f)
            f.write('\n')


    results = custom_evaluate_functional_correctness(temp_samples_file, k, n_workers, timeout, problem_file)
    print(results)

    # Clean up the temporary file
    os.remove(temp_samples_file)


def main():
    fire.Fire(entry_point)


sys.exit(main())


