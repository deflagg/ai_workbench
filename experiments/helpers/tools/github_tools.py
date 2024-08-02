import os
from git import Repo, GitCommandError
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL

@tool
def create_file(filename, content=None):
  """Creates a new file with the given filename and optional content.

  Args:
      filename (str): The name of the file to create.
      content (str, optional): The content to write to the file. Defaults to None.
  """

  try:
      with open(filename, "w") as f:
          if content:
              f.write(content)
      print(f"File '{filename}' created successfully.")
  except IOError as e:
      print(f"An error occurred: {e}")

# Example usage
# filename = "my_new_file.txt"
# content = "This is the content of my new file!"

# create_file(filename, content)

@tool
def get_repo(repo_name, repo_dir):
    """
    Authenticates with GitHub, clones (if necessary), and returns the specified repository object.

    Args:
        repo_name (str): The full name of the repository (e.g., "deflagg/ArtificialNeuralNetworkStudy").
        repo_dir (str): The local directory where the repository should be cloned or accessed.

    Returns:
        git.Repo.Repo: The Git repository object if authentication is successful, None otherwise.
    """

    try:
        github_token = os.getenv("GITHUB_ACCESS_TOKEN")

        # Clone or open the repository
        if os.path.exists(repo_dir):
            repo = Repo(repo_dir)
            # Update credentials for existing repository
            repo.git.config('http.extraheader', f'Authorization: token {github_token}')

        else:
            repo = Repo.clone_from(
                f"https://{github_token}@github.com/{repo_name}.git",
                repo_dir,
            )

        print("Authentication successful!")
        return repo

    except GitCommandError as e:
        print(f"Authentication failed: {e}")
        return None

# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

if __name__ == "__main__":
    repo_name = "deflagg/ArtificialNeuralNetworkStudy"
    repo_dir = "pile_of_shit/bug_killer/ArtificialNeuralNetworkStudy"  # Local directory for the repo

    repo = get_repo(repo_name, repo_dir)
    
    filename = "pile_of_shit/bug_killer/example_code_file.txt"
    content = "This is the content of my new file!"
    create_file(filename, content)

    if repo:
        # Example: Get the latest commit
        latest_commit = repo.head.commit
        print(f"Latest commit message: {latest_commit.message}")
        print(f"Latest commit date: {latest_commit.committed_date}")
        
        # Pull latest changes
        repo.git.pull('origin', 'main')
        
        # ... (perform other Git operations like pulling, branching, committing, etc.)
    else:
        print("Failed to authenticate with GitHub.")
