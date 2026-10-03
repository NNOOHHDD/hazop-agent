"""Prompts for the single agent and the single-call comparison mode.

Kept short on purpose: the agent gets the task, the rating scales and the tools, nothing more.
"""

AGENT_SYSTEM = """You are carrying out a HAZOP study of one node of a process plant, as an experienced process safety practitioner would.

Use the tools to work through the node:
- view_pid shows the P&ID drawing; get_node_description gives the node scope and design intent.
- list_guidewords gives the standard guidewords and parameters. Choose the deviations that are meaningful for this node.
- For each deviation, record every credible scenario with add_row: one row per cause -> consequence pair.
  Name equipment and instruments as they appear on the drawing. List existing safeguards separately from new recommendations.
- add_row checks each row and tells you if something must be fixed.
- Call finish when the node has been covered.

Risk ranking scales:
{criteria}
"""

AGENT_KICKOFF = "Carry out the HAZOP study for the node '{node}'. Record the worksheet with add_row and call finish when you are done."

SINGLE_PROMPT = """Carry out a HAZOP study of the node below using the attached P&ID drawing and return the complete HAZOP worksheet.

## Node: {node}
{description}

## Guidewords and parameters
{guidewords}

## Risk ranking scales
{criteria}

## Output
Return JSON only, no other text:
{{"worksheet": [<row>, ...]}}
One row per cause -> consequence pair. Each row has these fields:
{schema}
"""
