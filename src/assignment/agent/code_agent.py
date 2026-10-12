"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        self.system_prompt = (
            "You are a coding agent tasked with fixing a software issue in a git "
            "repository. Use the execute tool to inspect the repository, edit "
            "files, and run relevant tests. Make focused changes that address "
            "the user's task and follow the repository's conventions. Use "
            "ordinary assistant text to report progress. The send_message tool "
            "is reserved for final submission and ends the run. When finished, "
            "create and verify the git patch produced by git diff, then call "
            "send_message with a concise summary of the changes and tests run."
        )
        system_information = json.dumps(
            {
                "machine": environment.machine,
                "release": environment.release,
                "system": environment.system,
                "version": environment.version,
            },
            indent=2,
        )
        self.system_prompt += (
            f"\n\n<system_information>\n{system_information}\n</system_information>"
        )
        self.task_prompt = self.task
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute commands, load skills, or validate a final submission."""

        observations = []
        for call in tool_calls:
            try:
                if self.finished:
                    raise ValueError("The task has already been submitted.")
                function = call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")

                if name == "execute":
                    content = format_tool_output(self.env.execute(**arguments))
                elif name == "invoke_skill":
                    skill_name = arguments["name"]
                    if not isinstance(skill_name, str) or skill_name not in self.skills:
                        raise ValueError(f"Unknown skill: {skill_name!r}")
                    content = self.skills[skill_name]["content"]
                elif name == "send_message":
                    summary = arguments["summary"]
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError("The summary must be a non-empty string.")
                    patch_result = self.env.execute("cat patch.txt")
                    if patch_result["returncode"] != 0:
                        raise ValueError(
                            "Cannot submit: patch.txt could not be read. Create "
                            "and verify it in the default working directory first."
                        )
                    patch = patch_result["output"]
                    if not patch.strip():
                        raise ValueError(
                            "Cannot submit: patch.txt is empty. Finish the fix, "
                            "then create and verify a non-empty patch first."
                        )
                    self.submitted_patch = patch
                    print(summary)
                    self.finished = True
                    content = "Patch submitted. Task finished."
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as exc:
                content = format_tool_output({"error": str(exc)})

            observations.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
        return observations
