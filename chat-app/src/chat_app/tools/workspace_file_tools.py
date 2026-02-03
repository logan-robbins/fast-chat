"""Utilities for file tools scoped to the agent workspace.

Provides factory function to create sandboxed file operation tools that
restrict all operations to a designated workspace directory. This prevents
agents from accessing files outside their allowed directory.

The module wraps LangChain's file management tools with path normalization
that automatically strips workspace prefixes and blocks path traversal
attempts (e.g., "../" patterns).

Supported operations:
- copy_file: Copy files within workspace
- delete_file: Delete files within workspace
- file_search: Search for files by pattern
- list_directory: List directory contents
- move_file: Move/rename files within workspace
- read_file: Read file contents
- write_file: Write/create files

Dependencies:
- langchain_community>=0.3.0: For file management tools

Last Grunted: 02/03/2026 03:45:00 PM PST
"""

from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.copy import CopyFileTool
from langchain_community.tools.file_management.delete import DeleteFileTool
from langchain_community.tools.file_management.file_search import FileSearchTool
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langchain_community.tools.file_management.move import MoveFileTool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.utils import FileValidationError
from langchain_community.tools.file_management.write import WriteFileTool


def create_workspace_file_tools(workspace: Path | str) -> Sequence[BaseTool]:
    """Create sandboxed file tools restricted to the specified workspace.

    Factory function that creates LangChain file management tools with
    automatic path normalization to ensure all operations stay within
    the workspace directory. Protects against path traversal attacks.

    Args:
        workspace (Path | str): Root directory for file operations.
            Will be created if it doesn't exist. Can be relative or absolute.

    Returns:
        Sequence[BaseTool]: Tuple of 7 file tools, each sandboxed to workspace:
            - WorkspaceCopyFileTool
            - WorkspaceDeleteFileTool
            - WorkspaceFileSearchTool
            - WorkspaceMoveFileTool
            - WorkspaceReadFileTool
            - WorkspaceWriteFileTool
            - WorkspaceListDirectoryTool

    Raises:
        FileValidationError: If a tool operation attempts to access a path
            outside the resolved workspace directory.

    Example:
        >>> tools = create_workspace_file_tools("/tmp/agent_workspace")
        >>> read_tool = [t for t in tools if "read" in t.name.lower()][0]
        >>> read_tool.invoke({"file_path": "data.txt"})  # Reads /tmp/agent_workspace/data.txt

    Security:
        - Absolute paths outside workspace are rejected
        - Relative paths are resolved within workspace
        - Workspace prefix is stripped to handle "artifacts/file.txt" as "file.txt"
        - Parent directory traversal (../) is blocked by path resolution

    Last Grunted: 02/03/2026 03:45:00 PM PST
    """

    workspace_path = Path(workspace)
    resolved_workspace = workspace_path.resolve()
    workspace_parts: Sequence[str] = tuple(
        part for part in workspace_path.parts if part not in ("", ".")
    )

    def _normalize_workspace_path(path: str) -> str:
        if not path:
            return path

        candidate = Path(path)

        if candidate.is_absolute():
            try:
                relative_candidate = candidate.resolve().relative_to(resolved_workspace)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise FileValidationError(
                    f"Path {path} is outside of the allowed directory {resolved_workspace}"
                ) from exc
            return str(relative_candidate)

        filtered_parts = [
            part for part in candidate.parts if part not in ("", ".")
        ]

        if workspace_parts and tuple(filtered_parts[: len(workspace_parts)]) == tuple(
            workspace_parts
        ):
            filtered_parts = filtered_parts[len(workspace_parts) :]

        if not filtered_parts:
            return "."

        return str(Path(*filtered_parts))

    class _WorkspacePathMixin:
        def get_relative_path(self, file_path: str):  # type: ignore[override]
            normalized_path = _normalize_workspace_path(file_path)
            return super().get_relative_path(normalized_path)  # type: ignore[misc]

    class WorkspaceCopyFileTool(_WorkspacePathMixin, CopyFileTool):
        pass

    class WorkspaceDeleteFileTool(_WorkspacePathMixin, DeleteFileTool):
        pass

    class WorkspaceFileSearchTool(_WorkspacePathMixin, FileSearchTool):
        pass

    class WorkspaceListDirectoryTool(_WorkspacePathMixin, ListDirectoryTool):
        pass

    class WorkspaceMoveFileTool(_WorkspacePathMixin, MoveFileTool):
        pass

    class WorkspaceReadFileTool(_WorkspacePathMixin, ReadFileTool):
        pass

    class WorkspaceWriteFileTool(_WorkspacePathMixin, WriteFileTool):
        pass

    tool_classes: Iterable[type[BaseTool]] = (
        WorkspaceCopyFileTool,
        WorkspaceDeleteFileTool,
        WorkspaceFileSearchTool,
        WorkspaceMoveFileTool,
        WorkspaceReadFileTool,
        WorkspaceWriteFileTool,
        WorkspaceListDirectoryTool,
    )

    return tuple(tool(root_dir=str(resolved_workspace)) for tool in tool_classes)


__all__ = ["create_workspace_file_tools"]
