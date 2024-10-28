import together
import tenacity
import time
import re


client = together.Together()

# from https://arxiv.org/pdf/2410.02089 C.4
#CODEGEN_PROMPT = """Write a solution to the following problem and make sure that it passes the tests:
#{problem}"""

CODEGEN_PROMPT = """Provide a Python solution for the following programming
question.
Your code should be enclosed in triple backticks like so: ```python YOUR CODE HERE ```. Use the backticks for your code only.

```python
{problem}
```"""


"""
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(f"Retrying after error: {retry_state.outcome.exception()}")
)
"""
def get_completion(prompt: str) -> list[str]:
    """Get completion from Llama with retry logic"""
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
    )
    # assume only a single completion choice
    texts = [c.message.content for c in response.choices]
    #print(texts[0])
    return texts[0]


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
        echo=True  # Return the prompt with the response
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
    test = test.replace("def check", "def test")
    return f"""{output}
import pytest
@pytest.fixture
def candidate():
    return {entry_point}
{test}
"""

if __name__ == "__main__":
    import datasets
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    data = datasets.load_dataset("evalplus/humanevalplus", split="test")
    url = "https://justinchiu--runtest-dev.modal.run"

    for x in data:
        prompt = CODEGEN_PROMPT.format(problem = x["prompt"])
        output = get_completion(prompt)

        x_prompt = x["prompt"].strip()
        x_solution = x["canonical_solution"].strip()
        full_solution = f"""```python
{x_prompt}

{x_solution}
```"""
        tokens, logprobs = get_logprobs(prompt, full_solution)
        solution_len = len(tokenizer.tokenize(full_solution))
        # for debugging
        #completion_tokens = tokens[-solution_len:]
        logprob1 = sum(logprobs[-solution_len:])

        code = re.findall(r"```python\n(.*?)\n```", output, flags=re.MULTILINE|re.DOTALL)[0]

        request_code = convert_example(code, x["entry_point"], x["test"])
        response = requests.post(url, json={"codes": [request_code]})
        reports = response.json()

        for report in reports:
            if "failed" in report["summary"]:
                report_output = report["tests"][0]["call"]["longrepr"]
                out = re.findall(r"out = .*", report_output)[0]
                exp = re.findall(r"exp = .*", report_output)[0]
                import pdb; pdb.set_trace()
