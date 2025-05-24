from agents import function_tool
import streamlit as st
import os
import re


@function_tool
def list_directory_contents(path: str) -> str:
    """
    Generates a formatted tree view of the specified directory path and its contents.

    This function creates a visual representation of a directory structure, starting from 
    the root and displaying the full path hierarchy leading to the target directory. 
    It then lists all non-hidden files and subdirectories contained within that directory 
    in a tree-like format, similar to the Unix `tree` command.

    Hidden files and directories (those starting with a dot) are excluded from the output 
    to keep the view clean and focused on primary project files.

    Args:
        path (str): The absolute or relative path to the target directory. 
            The function will display the hierarchy from the filesystem root (or 
            provided base path) down to this directory, followed by its immediate contents.

    Returns:
        str: A formatted string representing the directory structure. 
            The output includes:
            - The full path hierarchy to the target directory.
            - A list of all non-hidden files and subdirectories within that directory.
            Each level is indented and connected using ASCII symbols to mimic a tree view.

    Notes:
        - Hidden files and folders (names starting with `.`) are ignored.
        - The function does **not** recursively expand subdirectories beyond the target directory.
        - The directory contents are sorted alphabetically for consistent output.
        - If the specified directory does not exist or cannot be accessed, an exception will be raised.
        - Designed for readability in terminal or text-based outputs.
        - Output is truncated to a maximum of 50,000 characters.

    Example:
        >>> list_directory_contents("/home/user/project/auth")

        Output:
        home/
        └── user/
            └── project/
                └── auth/
                    ├── routes.py
                    ├── utils.py
                    └── oauth/

    If the directory is empty:
        >>> list_directory_contents("/home/user/empty_folder")
        home/
        └── user/
            └── empty_folder/

    Raises:
        PermissionError: If access to the directory is denied.
    """

    character_limit = 50000

    # Normalize path for the OS
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    
    # Handle root directory differently based on OS
    if os.path.isabs(path):
        if os.name == 'nt':  # Windows
            root = parts[0] + os.sep
            hierarchy = parts[1:]
        else:  # Unix-like
            root = os.sep
            hierarchy = parts[1:]
    else:
        # Relative path
        root = parts[0] + os.sep if parts[0] else "."
        hierarchy = parts[1:]

    lines = [root]

    for depth, folder in enumerate(hierarchy):
        if not folder:  # Skip empty parts
            continue
        indent = "    " * depth
        lines.append(f"{indent}└── {folder}{os.sep}")

    # Calculate indent for contents
    base_indent = "    " * len([p for p in hierarchy if p])

    try:
        entries = [e for e in os.listdir(path) if not e.startswith(".")]
    except PermissionError:
        lines.append(base_indent + "    └── [permission denied]")
        return "\n".join(lines)[:character_limit]

    entries.sort(key=lambda e: e.lower())

    for i, name in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        full = os.path.join(path, name)
        suffix = os.sep if os.path.isdir(full) else ""
        lines.append(f"{base_indent}{connector}{name}{suffix}")

    return "\n".join(lines)[:character_limit]


@function_tool
def load_file_contents(path: str, line_start: int, line_end: int|None) -> str:
    """
    Reads and returns the contents of a file as a string, with optional support for reading 
    a specific range of lines.

    This function opens a text file and returns its content as a single string. It allows 
    you to specify a starting and ending line number to read only a subset of the file, 
    which is useful for handling large files or when only a specific section is needed. 
    To prevent excessive memory usage, the returned content is capped at 50,000 characters.

    Args:
        path (str): The absolute or relative path to the file to be read.
        line_start (int, optional): The 0-based index of the first line to read (inclusive).
            Defaults to 0, meaning reading starts from the beginning of the file.
        line_end (int, optional): The 0-based index of the last line to read (inclusive).
            If set to None, the function reads until the end of the file. Defaults to None.

    Returns:
        str: The requested file content as a single string, limited to 50,000 characters.
            Returns "No text was returned." if the specified line range results in empty content.
            If the file does not exist, cannot be read, or another error occurs,
            a descriptive error message string is returned instead (e.g., "Error reading file: ...").

    Notes:
        - This function reads files in text mode with UTF-8 encoding.
        - `line_start` is 0-based index. `line_end` is 0-based index, and the line at `line_end` is included.
        - If `line_end` is specified and is less than `line_start`, the function will return "No text was returned." (after internal slicing results in empty list).
        - Binary files or files with incompatible encoding may cause errors and result in an error string return.
        - If the selected lines exceed 50,000 characters, the output will be truncated.
        - Hidden files can be read as long as a valid path is provided.
        - By default (`line_start=0`, `line_end=None`), the entire file content (up to the character limit) is returned.

    Example:
        >>> load_file_contents("repository/auth/routes.py", line_start=10, line_end=20)
        'def login_user(request):\n    # Logic for user login...\n    ...'

        >>> load_file_contents("repository/README.md") # Example now works with optional line_start=0
        '# Project Overview\nThis repository handles...'

    If an error occurs:
        >>> load_file_contents("nonexistent/file.txt")
        'Error reading file: [Errno 2] No such file or directory: ...'
    """

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    contents = lines[line_start:(line_end + 1) if line_end else None]

    character_limit = 50000

    try:
        formatted_results = ''.join(contents)[:character_limit]
        return formatted_results or f"No text was returned."
    except Exception as e:
        return f"Error reading file: {e}"


@function_tool
def perform_string_search(query: str, path: str) -> str:
    """
    Performs a semantic search across all currently indexed repository content.

    This function leverages vector-based semantic search to identify relevant code snippets,
    documentation, or configuration details based on a natural language query. Unlike exact
    string matching, semantic search can understand contextual meaning, making it ideal for
    nuanced or complex queries where keywords alone may not suffice.

    It searches across all pre-indexed content (such as source code, markdown docs, or config files)
    and retrieves the top matches ranked by relevance. This is particularly useful for questions
    like "How is authentication handled?" or "Where is the database connection configured?"
    where the answer may span different terms or concepts.

    Args:
        query (str): A natural language query or keyword describing the information
            you're seeking within the codebase or documentation. This can be a question
            (e.g., "How are tokens refreshed?") or a topic (e.g., "database initialization").

    Returns:
        str: A formatted string containing the most relevant code snippets, documentation
            excerpts, or configuration fragments that match the query. Each result typically
            includes contextual information such as file paths or sections for traceability.

    Notes:
        - The quality of results depends on the comprehensiveness of the indexed data and
          the clarity of the query.
        - Files must be pre-indexed in the vector database before they can be searched.
        - This function is best suited for conceptual or broad searches. For precise symbol
          lookups (e.g., function names), consider using an exact string search instead.
        - The search is performed across *all* indexed content; it is not possible to restrict the search to a subdirectory using this tool.
        - Results are limited to the top 5 most relevant matches.

    Example:
        >>> perform_semantic_search("How is user authentication implemented?")
        'repository/auth/routes.py: Defines login and token refresh logic...\n\n
         repository/docs/authentication.md: Describes OAuth2 flow...\n\n
         ...' # Showing potentially multiple results
    """

    search_results = []
    for root, _, files in os.walk(path):
        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines):
                        if query in line:
                            pattern = r'\b' + query + r'\b'

                            if re.search(pattern, line):
                                metadata = '\n'.join((
                                    f"file_path: {file_path}",
                                    f"line_number: {i + 1}",
                                    f"line_content: {line.strip()}"
                                ))
                                search_results.append(metadata)
                except:
                    pass

    search_results_limit = 100
    formatted_results = '\n\n'.join(search_results[:search_results_limit])
    return formatted_results or f"No exact matches found for query: {query}"


@function_tool
def perform_semantic_search(query: str) -> str:
    """
    Performs a semantic search across all currently indexed repository content.

    This function leverages vector-based semantic search to identify relevant code snippets,
    documentation, or configuration details based on a natural language query. Unlike exact
    string matching, semantic search can understand contextual meaning, making it ideal for
    nuanced or complex queries where keywords alone may not suffice.

    It searches across all pre-indexed content (such as source code, markdown docs, or config files)
    and retrieves the top matches ranked by relevance. This is particularly useful for questions
    like "How is authentication handled?" or "Where is the database connection configured?"
    where the answer may span different terms or concepts.

    Args:
        query (str): A natural language query or keyword describing the information
            you're seeking within the codebase or documentation. This can be a question
            (e.g., "How are tokens refreshed?") or a topic (e.g., "database initialization").

    Returns:
        str: A formatted string containing the most relevant code snippets, documentation
            excerpts, or configuration fragments that match the query. Each result typically
            includes contextual information such as file paths or sections for traceability.

    Notes:
        - The quality of results depends on the comprehensiveness of the indexed data and
          the clarity of the query.
        - Files must be pre-indexed in the vector database before they can be searched.
        - This function is best suited for conceptual or broad searches. For precise symbol
          lookups (e.g., function names), consider using an exact string search instead.
        - The search is performed across *all* indexed content; it is not possible to restrict the search to a subdirectory using this tool.
        - Results are limited to the top 5 most relevant matches.

    Example:
        >>> perform_semantic_search("How is user authentication implemented?")
        'repository/auth/routes.py: Defines login and token refresh logic...\n\n
         repository/docs/authentication.md: Describes OAuth2 flow...\n\n
         ...' # Showing potentially multiple results
    """

    retriever = st.session_state.index.as_retriever(similarity_top_k=5)
    response = retriever.retrieve(query)
    formatted_response = '\n\n'.join([i.text for i in response])
    return formatted_response