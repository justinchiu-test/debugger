import together
import anthropic
import time
import re
import json
import ast
import math

from debugger.prompts import CODEGEN_PROMPT, REPAIR_PROMPT, PRINT_PROMPT, REPAIR_AFTER_PRINT_PROMPT
from debugger.exec import run_python_code

client = together.Together()
aclient = anthropic.Anthropic()

# Template for constructing tests
TEST_PREFIX = """
import numpy as np


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def assertion(out, exp, atol):
    exact_match = out == exp

    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if not exact_match and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert exact_match
"""

TEST_SINGLE = """
def test{i}(candidate):
    assertion(candidate(*{input}), {result}, 0)
"""


def get_completion(prompt: str) -> list[str]:
    """Get greedy completion from Llama"""
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=2048,
        temperature=0.,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
    )
    # assume only a single completion choice
    texts = [c.message.content for c in response.choices]
    # print(texts[0])
    return texts[0]


def get_acompletion(prompt: str) -> list[str]:
    """Get greedy completion from claude"""
    message = aclient.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0.,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    )
    return message.content[0].text


def get_logprobs(prompt: str, completion: str) -> tuple[list[str], list[float]]:
    """
    Get log probabilities for each token in the given text.

    Args:
        text: The text to get log probs for
        context: Optional context/prefix before the text

    Returns:
        Tuple of (tokens, log_probs)
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": completion,
            },
        ],
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=0,  # We don't want to generate new tokens
        temperature=0,
        logprobs=True,  # Enable log probabilities
        top_k=1,
        echo=True,  # Return the prompt with the response
    )

    # Extract logprobs from response
    logprobs_info = response.prompt[0].logprobs

    # Get tokens and their log probabilities
    tokens = logprobs_info.tokens
    log_probs = logprobs_info.token_logprobs

    # Remove None values that appear for the first token
    tokens = [t for t, lp in zip(tokens, log_probs) if lp is not None]
    log_probs = [lp for lp in log_probs if lp is not None]

    return tokens, log_probs


def convert_example(output, entry_point, test):
    inputs = ast.literal_eval(re.findall(r"inputs = (.*)", test)[0])
    if "results" in test:
        results = ast.literal_eval(re.findall(r"results = (.*)", test)[0])
        test_text = "\n".join(
            [
                TEST_SINGLE.format(i=i, input=repr(input), result=repr(result))
                for i, (input, result) in enumerate(zip(inputs, results))
            ]
        )
    else:
        results = [f"ref_func({repr(*input)})" for input in inputs]
        test_text = "\n".join(
            [
                TEST_SINGLE.format(i=i, input=repr(input), result=repr(result))
                for i, (input, result) in enumerate(zip(inputs, results))
            ]
        )
    test_string = f"""{output}
import pytest
@pytest.fixture
def candidate():
    return {entry_point}

{TEST_PREFIX}

{test_text}
"""
    return test_string, inputs, results


if __name__ == "__main__":
    import datasets
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    data = datasets.load_dataset("evalplus/humanevalplus", split="test")
    url = "https://justinchiu--runtest-dev.modal.run"

    for x in data:
        prompt = CODEGEN_PROMPT.format(problem=x["prompt"])
        output = get_completion(prompt)

        x_prompt = x["prompt"].strip("\n")
        x_solution = x["canonical_solution"].strip("\n")
        full_solution = f"""```python
{x_prompt}

{x_solution}
```"""
        tokens, logprobs = get_logprobs(prompt, full_solution)
        solution_len = len(tokenizer.tokenize(full_solution))
        # for debugging
        # completion_tokens = tokens[-solution_len:]
        logprob1 = sum(logprobs[-solution_len:])

        code = re.findall(
            r"```python\n(.*?)\n```", output, flags=re.MULTILINE | re.DOTALL
        )[0]

        request_code, inputs, results = convert_example(
            code, x["entry_point"], x["test"]
        )
        response = requests.post(url, json={"codes": [request_code]})
        reports = response.json()

        print("reports:", len(reports))
        # should only be one report (test suite) with many tests
        report = reports[0]

        # gather failed test idxs
        test_idxs = []
        # only care about programs with >= 1 failed test cases
        if "failed" in report["summary"]:
            print("tests:", len(report["tests"]))
            for i, test in enumerate(report["tests"]):
                # find failed test case
                if test["outcome"] == "failed":
                    test_idxs.append(i)

        for idx in test_idxs:
            test = report["tests"][idx]

            # re-execute the code b/c it's annoying to parse in pytest
            print("outcome:", test["outcome"])
            input = inputs[i]  # in list form
            result = results[i]  # expected
            exec_output, exec_err = run_python_code(code + f"\nprint({x['entry_point']}(*{input}))")

            repair_prompt = REPAIR_PROMPT.format(
                code=code, input=input, output=exec_output, expected_output=result
            )

            tokens, logprobs = get_logprobs(repair_prompt, full_solution)
            logprob2 = sum(logprobs[-solution_len:])

            lines = repair_prompt.splitlines()
            ts, ls = get_logprobs("\n".join(lines[:-6] + [lines[-1]]), full_solution)
            logprob3 = sum(ls[-solution_len:])

            print_prompt = PRINT_PROMPT.format(
                code=code, input=input, output=exec_output, expected_output=result
            )
            print_output = get_acompletion(print_prompt)
            print_code = re.findall(
                r"```python\n(.*?)\n```", print_output, flags=re.MULTILINE | re.DOTALL
            )[0]

            print_out, print_err = run_python_code(print_code + f"\nprint({x['entry_point']}(*{input}))")

            # format print_code by splicing prints with printed
            print_code_lines = print_code.splitlines()
            print_out_lines = print_out.splitlines()
            lines = []
            print_idx = 0
            for line in print_code_lines:
                if "print" in line:
                    lines.append(line + "# " + print_out_lines[print_idx])
                    print_idx += 1
                else:
                    lines.append(line)
            printed_code = "\n".join(lines)

            repair_after_print_prompt = REPAIR_AFTER_PRINT_PROMPT.format(
                code=printed_code, input=input, output=exec_output, expected_output=result
            )

            # claude repairs
            #repair_claude = get_acompletion(repair_prompt)
            #repair_after_print_claude = get_acompletion(repair_after_print_prompt)
            # get llama repairs
            repair_claude = get_completion(repair_prompt)
            repair_after_print_claude = get_completion(repair_after_print_prompt)

            repair_code = re.findall(
                r"```python\n(.*?)\n```", repair_claude, flags=re.MULTILINE | re.DOTALL
            )[0]
            repair_code2 = re.findall(
                r"```python\n(.*?)\n```", repair_after_print_claude, flags=re.MULTILINE | re.DOTALL
            )[0]

            repair_req, inputs, results = convert_example(
                repair_code, x["entry_point"], x["test"]
            )
            repair_response = requests.post(url, json={"codes": [repair_req]})
            repair_reports = repair_response.json()
            repair_req2, inputs, results = convert_example(
                repair_code2, x["entry_point"], x["test"]
            )
            repair_response2 = requests.post(url, json={"codes": [repair_req2]})
            repair_reports2 = repair_response2.json()

            import pdb; pdb.set_trace()
