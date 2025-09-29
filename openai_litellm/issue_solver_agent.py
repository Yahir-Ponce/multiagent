

from dotenv import load_dotenv
import os, requests
from bs4 import BeautifulSoup
from agents import Agent, Runner, set_tracing_disabled, function_tool, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

# --- Load environment variables ---
load_dotenv()
set_tracing_disabled(True)
API_KEY = os.getenv('AZURE_API_KEY')

# --- function_tool: fetch_url -> scrapes a URL using BS4 ---
@function_tool
def fetch_url(url: str) -> str:
    """
    Scrapes the content of a URL and returns only plain text.
    
    Args:
        url (str): URL provided by the user
    
    Returns:
        str: Plain text from the page (max 8000 characters)
    """
    try:
        print("[Looking for info]")
        r = requests.get(url, timeout=15)    # 15s timeout to avoid hanging
        r.raise_for_status()                 # Raises exception if status != 200
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove irrelevant tags for reading
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract clean text
        text = soup.get_text(separator=" ", strip=True)
        return text[:8000]  # Limit to 8k characters (to avoid saturating the model)
    except Exception as e:
        return f"Error fetching {url}: {e}"


# --- Validator: evaluates the patch, does NOT generate code ---
validator_agent = Agent(
    name="Validator",
    instructions="""
You are a Patch Validator.

**Input:** A unified git diff produced by Engineer and the original issue summary.

**Your Task:** Check the following:
- Applies cleanly to mentioned files (static reasoning; no execution)
- Matches the issue scope (no unnecessary collateral changes)
- Probable risks/regressions

**Output:** JSON ONLY with the following keys:
```json
{
  "applies_cleanly": true|false,
  "matches_issue": true|false,
  "risks": ["..."],
  "gaps": ["..."],
  "test_plan": ["pytest ...", "npm test", "script/ci"],
  "verdict": "approve|revise"
}
```

**Rules:**
- Do NOT output the diff
- Do NOT modify the patch
- Only provide validation analysis

**IMPORTANT: After providing your JSON validation, immediately transfer back to Orchestrator.**
""",
    model=LitellmModel(
        model="azure/gpt-5",
        api_key=API_KEY,
    ),
    model_settings=ModelSettings(include_usage=True),
)


# --- Engineer / Solver: returns ONLY the patch in git format ---
solver_agent = Agent(
    name="Engineer",
    instructions="""
        You are a Git Patch Generator.

        **Input:** A normalized analysis of a GitHub issue (and optional research context).

        **CRITICAL OUTPUT REQUIREMENT:**
        Your response must be ONLY a valid unified git diff. Nothing else.

        Start with "diff --git" and end with the last patch line.
        Do NOT include:
        - Explanations or commentary
        - Markdown code fences (no ```)
        - JSON formatting
        - Headers like "Here's the patch:"
        - Summaries

        **Required format:**
        diff --git a/<path> b/<path>
        index <oldsha>..<newsha> 100644
        --- a/<path>
        +++ b/<path>
        @@ -<start>,<len> +<start>,<len> @@ <optional context>
        -old line
        +new line

        **Rules:**
        - Return ONLY raw patch text
        - Keep trailing newline at end of file
        - Must be applicable with `git apply` or `patch -p1`
        - First line MUST start with "diff --git"

        **IMPORTANT: After outputting the diff, immediately transfer back to Orchestrator.**
    """,
    model=LitellmModel(
        model="azure/gpt-5",
        api_key=API_KEY,
    ),
    model_settings=ModelSettings(include_usage=True)
)


# --- Researcher / Explorer ---
researcher_agent = Agent(
    name="Explorer",
    instructions="""
        Role: Quick research and context gathering.

        Task:
        - Investigate similar patterns, probable root causes, and signals to check.
        - If you rely on a source, mention it briefly (e.g., "source: <url>" or "source: internal").
        - Be concise, technical, and actionable.
        - Do not invent links.

        Handoff:
        - After finishing, return control to the Solver Agent.
    """,
    model=LitellmModel(
        model="azure/gpt-5",
        api_key=API_KEY,
    ),
    model_settings=ModelSettings(include_usage=True),
    handoffs=[solver_agent]
)


# --- Analyzer ---
analyzer_agent = Agent(
    name="Analyzer",
    instructions="""
        **Role:** Issue analyzer and normalizer.

        **Task:** Parse and categorize the issue; extract:
        - Type (bug/feature/regression/doc)
        - Affected components/paths
        - Severity (low/med/high/critical)
        - Mentioned technologies
        - Symptoms and reproduction steps (if available)

        **IMPORTANT: After providing your analysis, ALWAYS immediately transfer back to Researhcer Agent.**

        **Rules:**
        - Be objective and factual
        - Extract information directly from the issue text
        - Don't add assumptions not present in the original issue
    """,
    model=LitellmModel(
        model="azure/gpt-5",
        api_key=API_KEY,
    ),
    model_settings=ModelSettings(include_usage=True),
    tools=[fetch_url],
    handoffs=[researcher_agent]  # Solo puede regresar al Orchestrator
)


# --- Orchestrator: returns VERBATIM the diff from Solver ---
orchestrator_agent = Agent(
    name="Orchestrator",
    instructions="""
        You orchestrate four agents to resolve a plain-text GitHub issue.

        Workflow:
        1) Hand off to Analyzer with the raw issue text.
        2) Review the analysis; if the type is bug/regression or open questions exist → hand off to Explorer.
        3) Hand off to Engineer with the analysis (and Explorer’s context if available), requesting a unified git diff patch.
        4) Hand off to Validator with the patch and the original analysis, requesting validation of risks and scope.
        5) Final step: output only the raw git diff from Engineer (verbatim).

        Rules:
        - The final output must start with "diff --git".
        - Do not include validation results, commentary, or wrappers.
        - No Markdown fences or "Here is the patch:" text.
        - Return ONLY raw patch text
        - Keep trailing newline at end of file
        - Must be applicable with `git apply` or `patch -p1`
        - First line MUST start with "diff --git"
    """,
    model=LitellmModel(
        model="azure/gpt-5",
        api_key=API_KEY,
    ),
    model_settings=ModelSettings(include_usage=True, parallel_tool_calls=False),
    handoffs=[analyzer_agent]
)


def normalize_text_to_one_line(text: str) -> str:
    """
    Replace tabs with \r and newlines with \n, 
    flattening the text into a single line.
    """
    # Reemplazar tabuladores por el literal \r
    text = text.replace("\r", "\\r")
    # Reemplazar saltos de línea por el literal \n
    text = text.replace("\n", "\\n")
    return text


# --- Example execution ---
if __name__ == "__main__":
    issue_text = input("Paste a GitHub issue here:\n")
    print("Processing...\n")
    r = Runner.run_sync(
        orchestrator_agent,
        input=[{
            "role": "user",
            "content": issue_text
        }]
    )
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\nFinal output (git diff):\n")
    
    print(normalize_text_to_one_line(r.final_output))