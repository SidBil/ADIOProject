/**
 * Session path geometry — the pure derivation of the trail.
 *
 * Per PRD `home & session path/03_navigation_shell.md` §5, the entire trail is a
 * pure function of ONE durable quantity: how many sessions the child has
 * completed. Node states (completed/current/locked), the Stage-1/Stage-2 rhythm,
 * and milestone (goal) positions are all derived here — never persisted per node.
 *
 * Two independent axes (02_session_path.md §6):
 *   - Stage axis:  Stage 1 vs Stage 2 (recall) — every 3rd node is Stage 2.
 *   - Goal axis:   ordinary vs milestone node — the 5, 11, 18, 26… cadence.
 * A single node can be both.
 */

export type Stage = "stage1" | "stage2";
export type NodeState = "completed" | "current" | "locked";

export interface PathNode {
  /** 1-based node number == the child's Nth session on the trail. */
  number: number;
  stage: Stage;
  isMilestone: boolean;
  state: NodeState;
}

/**
 * Stage cadence: a fixed 2:1 pattern — S1, S1, S2, repeating (02_session_path.md
 * §6). Every 3rd node (1-based) is the Stage-2 recall node. Not configurable.
 */
export function stageForNumber(nodeNumber: number): Stage {
  return nodeNumber % 3 === 0 ? "stage2" : "stage1";
}

/**
 * Backend-facing helper: the mode to stamp on the *next* session given how many
 * the child has already completed. The next session is node `completedCount + 1`.
 */
export function modeForNextSession(completedCount: number): Stage {
  return stageForNumber(completedCount + 1);
}

/**
 * Milestone node numbers — the goals the trail climbs toward
 * (`gamification/03_milestones.md` §5): first at 5, then gaps that widen but are
 * capped so the trail never looks stuck. Documented sequence: 5, 11, 18, 26, 34,
 * 42… (gaps 6, 7, 8, 8, 8 — cap 8). Finite set (~8–10 milestones).
 */
export const MILESTONE_COUNT = 10;
const FIRST_MILESTONE = 5;
const MAX_GAP = 8;

export function milestoneNodes(count: number = MILESTONE_COUNT): number[] {
  const out: number[] = [FIRST_MILESTONE];
  let n = FIRST_MILESTONE;
  for (let i = 1; i < count; i++) {
    const gap = Math.min(6 + (i - 1), MAX_GAP); // 6, 7, 8, 8, …
    n += gap;
    out.push(n);
  }
  return out;
}

const MILESTONES = milestoneNodes();
const MILESTONE_SET = new Set(MILESTONES);

export function isMilestone(nodeNumber: number): boolean {
  return MILESTONE_SET.has(nodeNumber);
}

/**
 * The current goal node: the smallest milestone at or beyond the child's current
 * position. After the finite milestone set is exhausted, there is no fixed goal
 * (open question, north_star §6) — returns null and the trail renders a short
 * open run instead.
 */
export function nextMilestone(completedCount: number): number | null {
  const currentNumber = completedCount + 1;
  for (const m of MILESTONES) {
    if (m >= currentNumber) return m;
  }
  return null;
}

/**
 * Build the rendered window of the trail from the completed-session count.
 *
 * Render window (02_session_path.md §7): completed nodes behind (scrollable
 * history), the current lit node, and upcoming nodes THROUGH the next milestone.
 * Nodes beyond the next milestone are not rendered until it's reached.
 */
export function buildPath(completedCount: number): PathNode[] {
  const currentNumber = completedCount + 1;
  const goal = nextMilestone(completedCount);
  // Render through the goal; past the finite milestone set, show a short open run.
  const lastNumber = goal ?? currentNumber + 3;

  const nodes: PathNode[] = [];
  for (let n = 1; n <= lastNumber; n++) {
    let state: NodeState;
    if (n < currentNumber) state = "completed";
    else if (n === currentNumber) state = "current";
    else state = "locked";

    nodes.push({
      number: n,
      stage: stageForNumber(n),
      isMilestone: isMilestone(n),
      state,
    });
  }
  return nodes;
}
